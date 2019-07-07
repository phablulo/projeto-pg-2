#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <thread>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include "shader.hpp"
using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;
using namespace glm;

struct CalibrateResult {
  Mat intrinsic;
  Mat distCoeffs;
  vector<Point3f> objectPoints;
  vector<Mat> rvecs;
  vector<Mat> tvecs;
};
struct SharedResult {
  Mat image;
  vector<Point2f> corners;
  vector<Point2f> ipoints;
  bool isReady;
  SharedResult(): isReady(false) {}
};
SharedResult sr;

CalibrateResult calibrate(VideoCapture &cap) {
  int numBoards = 20;
  int numCornersHor = 7;
  int numCornersVer = 7;
  int numSquares = numCornersHor * numCornersVer;
  Size boardSize = Size(numCornersHor, numCornersVer);
 
  vector<vector<Point3f>> objectPoints;
  vector<vector<Point2f>> imagePoints;
  vector<Point2f> corners;
  int success = 0;
  Mat image;
  Mat imageGray;
  vector<Point3f> obj;

  for (int i = 0; i < numSquares; ++i) {
    obj.push_back(Point3f(i/numCornersHor, i%numCornersHor, 0.0f));
  }
  while (success < numBoards) {
    cap >> image;
    cvtColor(image, imageGray, COLOR_BGR2GRAY);
    bool found = findChessboardCorners(imageGray, boardSize, corners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FILTER_QUADS);
    if (found) {
      cout << "Found! " << ++success*100/numBoards << '%' << endl;
      cornerSubPix(imageGray, corners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.1));
      drawChessboardCorners(image, boardSize, corners, found);
      imagePoints.push_back(corners);
      objectPoints.push_back(obj);
    }
    sr.image = image;
    if (waitKey(1) == 27) {
      exit(0);
    }
  }

  Mat intrinsic(3, 3, CV_32FC1);
  Mat distCoeffs;
  vector<Mat> rvecs;
  vector<Mat> tvecs;
  intrinsic.ptr<float>(0)[0] = 1;
  intrinsic.ptr<float>(1)[1] = 1;
  calibrateCamera(objectPoints, imagePoints, image.size(), intrinsic, distCoeffs, rvecs, tvecs);

  CalibrateResult cr;
  cr.intrinsic = intrinsic;
  cr.distCoeffs = distCoeffs;
  cr.objectPoints = obj;
  cr.rvecs = rvecs;
  cr.tvecs = tvecs;

  return cr;
}
int track(VideoCapture &cap, Mat &source, CalibrateResult &cr) {
  Mat sourceGray; 
  cvtColor(source, sourceGray, COLOR_BGR2GRAY);
  Ptr<SURF> detector = SURF::create(200);
  vector<KeyPoint> keypoints_1, keypoints_2;
  Mat descriptors_1, descriptors_2;
  detector->detectAndCompute(sourceGray, noArray(), keypoints_1, descriptors_1);
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
  vector<vector<DMatch>> matches;
  vector<DMatch> goodMatches;
  const float ratio_thresh = 0.7f;
  Mat imagem;
  Mat image;
  Mat imageGray;

  while (true) {
    cap >> imagem;
    goodMatches.clear();
    matches.clear();
    undistort(imagem, image, cr.intrinsic, cr.distCoeffs);
    cvtColor(image, imageGray, COLOR_BGR2GRAY);
    
    detector->detectAndCompute(imageGray, noArray(), keypoints_2, descriptors_2);
    matcher->knnMatch(descriptors_1, descriptors_2, matches, 2);
    if (matches.size() > 0) {
      double minDist = 100000;
      for(int i = 0; i < descriptors_1.rows; ++i) {
        if (matches[i][0].distance < ratio_thresh * matches[i][1].distance) {
          goodMatches.push_back(matches[i][0]);
        }
      }
      Mat draw;
      try {
        drawMatches(source, keypoints_1, image, keypoints_2, goodMatches, draw, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
      } catch (...) {}
      image = draw;
    }
    
    sr.image = image;
    if (waitKey(1) == 27) {
      break;
    }
  } 
  return 0;
}
int simpleVideo(VideoCapture &cap, CalibrateResult &cr) {
  Mat imagem;
  Mat image;
  Mat imageGray;
  Size boardSize = Size(7, 7);
  vector<Point2f> corners;
  while (true) {
    cap >> imagem;
    undistort(imagem, image, cr.intrinsic, cr.distCoeffs);
    cvtColor(image, imageGray, COLOR_BGR2GRAY);
    bool found = findChessboardCorners(imageGray, boardSize, corners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FILTER_QUADS);
    if (found) {
      cornerSubPix(imageGray, corners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.1));
      drawChessboardCorners(image, boardSize, corners, found);
    }
    sr.image = image;
    if (waitKey(1) == 27) {
      break;
    }
  }
  return 0;
}
void drawObject(Mat &image, vector<Point2f> &corners, vector<Point2f> &ipoints) {
	vector<Point> floor;
	vector<vector<Point>> contours;
	
	for (int i = 0; i < 4; ++i) {
		floor.push_back(Point((double)ipoints[i].x, (double)ipoints[i].y));
	}
	contours.push_back(floor);
  drawContours(image, contours, -1, Scalar(0, 255, 0), -3);
	for (int i = 0; i < 4; ++i) {
		int j = 4 + i;
		line(image, ipoints[i], ipoints[j], Scalar(255, 0, 0), 3);
	}
	for (int i = 0; i < 4; ++i) {
		int j = 4 + i;
		floor[i] = Point((double)ipoints[j].x, (double)ipoints[j].y);
	}
	contours[0] = floor;
  drawContours(image, contours, -1, Scalar(0, 0, 255), 3);
	

	// coordenadas:
  // Point2f corner = corners[0];
  // line(image, corner, ipoints[0], Scalar(255, 0, 0), 5);
  // line(image, corner, ipoints[1], Scalar(0, 255, 0), 5);
  // line(image, corner, ipoints[2], Scalar(0, 0, 255), 5);
}
int poseEstimation(VideoCapture &cap, CalibrateResult &cr) {
  Mat imagem;
  Mat image;
  Mat imageGray;
  Size boardSize = Size(7, 7);
  vector<Point2f> corners;
  Mat rvec;
  Mat tvec;
  vector<Point2f> ipoints; 
  float data[24] = {0,0,0, 0,5,0, 5,5,0, 5,0,0, 0,0,5, 0,5,5, 5,5,5, 5,0,5};
  Mat axis(8, 3, CV_32F, data);

  while (true) {
    cap >> imagem;
    undistort(imagem, image, cr.intrinsic, cr.distCoeffs);
    cvtColor(image, imageGray, COLOR_BGR2GRAY);
    bool found = findChessboardCorners(imageGray, boardSize, corners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FILTER_QUADS);
    if (found) {
      cornerSubPix(imageGray, corners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.1));
      // encontra os vetores de rotação e translação
      solvePnPRansac(cr.objectPoints, corners, cr.intrinsic, cr.distCoeffs, rvec, tvec);
      // projeta-os no plano de imagem
      projectPoints(axis, rvec, tvec, cr.intrinsic, cr.distCoeffs, ipoints);
      // desenha
      // repassa dados pro SharedResult:
      sr.corners = corners;
      sr.ipoints = ipoints;
      sr.isReady = true;
      //drawObject(image, corners, ipoints);
    }
    else {
      sr.isReady = false;
    }
    sr.image = image;
    if (waitKey(1) == 27) {
      break;
    }
  }
  return 0;
}

void opencv() {
  VideoCapture cap(0);
  cap.set(CAP_PROP_FRAME_WIDTH, 640);
  cap.set(CAP_PROP_FRAME_HEIGHT, 480);
  CalibrateResult calibration;
  int status = 0;

  calibration = calibrate(cap);
  while (true) {
    if (status == 0) {
      simpleVideo(cap, calibration);
    }
    else if (status == 1) {
      Mat image = imread("imagem.jpg", 1); // imagem base
      if (image.empty()) {
        cout << "impossível ler a imagem pro tracking" << endl;
        exit(1);
      }
      track(cap, image, calibration);
    }
    else if (status == 2) {
      poseEstimation(cap, calibration);
    }
    status = (status + 1) % 3;
  }
  cap.release();
  exit(0);
}
// void opengl() {
//   glewExperimental = true; // Needed for core profile	
//   if (!glfwInit()) {
//     cout << "Erro ao iniciar GLFW" << endl;
//     exit(1);
//   }
//   glfwWindowHint(GLFW_SAMPLES, 4); // 4x antialiasing
//   glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); // We want OpenGL 3.3
//   glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
//   glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // We don't want the old OpenGL
// 
//   GLFWwindow* window;
//   window = glfwCreateWindow(640, 480, "Projeto PG", NULL, NULL);
//   if(window == NULL) {
//     cout << "Erro ao abrir janela do GLFW" << endl;
//     glfwTerminate();
//     exit(1);
//   }
//   glfwMakeContextCurrent(window);
//   glewExperimental=true; // Needed in core profile
//   if (glewInit() != GLEW_OK) {
//     cout << "Failed to initialize GLEW" << endl;
//     exit(1);
//   }
// 
// 	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
// 	GLuint textureID;
// 	glGenTextures(1, &textureID);
// 	// Bind to our texture handle
// 	glBindTexture(GL_TEXTURE_2D, textureID);
// 	GLenum inputColourFormat = GL_BGR;
// 	
// 	do {
// 		glClear(GL_COLOR_BUFFER_BIT);
// 		glMatrixMode(GL_MODELVIEW);
// 		glEnable(GL_TEXTURE_2D);
// 
//     glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, sr.image.cols , sr.image.rows, 0, inputColourFormat, GL_UNSIGNED_BYTE, sr.image.ptr());
//     glUniform1i(textureID, 0);
//     glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
//     glFlush();
// 	
// 		glBegin(GL_QUADS);
// 		glTexCoord2i(0, 0); glVertex2i(0,   0);
// 		glTexCoord2i(0, 1); glVertex2i(0,   480);
// 		glTexCoord2i(1, 1); glVertex2i(640, 480);
// 		glTexCoord2i(1, 0); glVertex2i(640, 0);
// 		glEnd();
// 
// 		glDeleteTextures(1, &textureID);
//     glDisable(GL_TEXTURE_2D);
// 
//     //this_thread::sleep_for(chrono::seconds(1));
//     cout << "isReady? " << sr.isReady << endl;
// 		//imshow("name", sr.image);
// 
// 		glfwSwapBuffers(window);
// 		glfwPollEvents();
// 	}
// 	while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(window) == 0);
// }
void opengl() {
	glewExperimental = true; // Needed for core profile
  if(!glfwInit()) {
		cout << "Falha ao iniciar GLFW" << endl;
    exit(1);
  }
	glfwWindowHint(GLFW_SAMPLES, 4); // 4x antialiasing
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); // We want OpenGL 3.3
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // We don't want the old OpenGL 
	GLFWwindow* window; // (In the accompanying source code, this variable is global for simplicity)
  window = glfwCreateWindow(649, 480, "Tutorial 01", NULL, NULL);
	if (window == NULL) {
		cout << "Falha ao abrir janela do GLFW" << endl;
		exit(1);
	}
	glfwMakeContextCurrent(window);
	glewExperimental = true;
	if (glewInit() != GLEW_OK) {
		cout << "Falha ao iniciar Glew" << endl;
		exit(1);
	}
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
	while (sr.image.empty()); // aguarda até ter imagem pra exibir
	cout << "Já existe imagem pra exibir" << endl;
	// cria objeto pra mostrar

  GLuint programID = LoadShaders("vshader.glsl", "fshader.glsl"); // carrega shaders
	GLuint VertexArrayID;
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);

	static const GLfloat g_vertex_buffer_data[] = {
		-1.0f, -1.0f, 0.0f,
		1.0f, -1.0f, 0.0f,
		0.0f,  1.0f, 0.0f,
	};
	GLuint vertexbuffer;
	glGenBuffers(1, &vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);
  GLuint textureID; // textura...
  glGenTextures(1, &textureID);

	// --- 
	do {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		// ---
		// desenhos acontecem aqui!
    glUseProgram(programID); // use o shader
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
		glDrawArrays(GL_TRIANGLES, 0, 3);
		glDisableVertexAttribArray(0);	

		// ---
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	while (glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS && glfwWindowShouldClose(window) == 0);
}

int main(int argc, char* argv[]) {
  thread opencv_t(opencv);
  opengl();
  return 0;
}
