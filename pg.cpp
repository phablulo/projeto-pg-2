#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

int main(int argc, char* argv[]) {
  int numBoards = 20;
  int numCornersHor = 7;
  int numCornersVer = 7;
  // printf("Número de quadrados horizontais: ");
  // scanf("%d", &numCornersHor);
  // printf("Número de quadrados verticais: ");
  // scanf("%d", &numCornersVer);
  // printf("Número de samples: ");
  // scanf("%d", &numBoards);

  int numSquares = numCornersHor * numCornersVer;
  Size boardSize = Size(numCornersHor, numCornersVer);
  

  // calibração
  VideoCapture cap(0);
  cap.set(CAP_PROP_FRAME_WIDTH, 640);
  cap.set(CAP_PROP_FRAME_HEIGHT, 480);
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
    bool found = findChessboardCorners(image, boardSize, corners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FILTER_QUADS);
    if (found) {
      cout << "Found! " << ++success*100/numBoards << '%' << endl;
      cornerSubPix(imageGray, corners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.1));
      drawChessboardCorners(image, boardSize, corners, found);
      imagePoints.push_back(corners);
      objectPoints.push_back(obj);
    }
    imshow("image", image);
    if (waitKey(1) == 27) {
      return 0;
    }
  }
  Mat intrinsic(3, 3, CV_32FC1);
  Mat distCoeffs;
  vector<Mat> rvecs;
  vector<Mat> tvecs;
  intrinsic.ptr<float>(0)[0] = 1;
  intrinsic.ptr<float>(1)[1] = 1;
  calibrateCamera(objectPoints, imagePoints, image.size(), intrinsic, distCoeffs, rvecs, tvecs);
  Mat undistorted;
  Mat undistortedGray;
  // calibração feita, meu querido!
  
  // agora vem o tracking
  Mat query = imread("imagem.jpg", IMREAD_GRAYSCALE); // imagem base
  if (query.empty()) {
    cout << "impossível ler a imagem pro tracking" << endl;
    return 1;
  }
  Ptr<SURF> detector = SURF::create(200);
  vector<KeyPoint> keypoints_1, keypoints_2;
  Mat descriptors_1, descriptors_2;
  detector->detectAndCompute(query, Mat(), keypoints_1, descriptors_1);
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
  vector<vector<DMatch>> matches;
  //vector<DMatch> matches;
  vector<DMatch> goodMatches;
  const float ratio_thresh = 0.7f;

  // pronto
  
  while (true) {
    cap >> image;
    goodMatches.clear();
    matches.clear();
    undistort(image, undistorted, intrinsic, distCoeffs);
    cvtColor(undistorted, undistortedGray, COLOR_BGR2GRAY);
    // tracking
    detector->detectAndCompute(undistortedGray, noArray(), keypoints_2, descriptors_2);
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
        drawMatches(query, keypoints_1, undistorted, keypoints_2, goodMatches, draw, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
      } catch (...) {}
      undistorted = draw;
    }
    // end
    imshow("image", undistorted);
    if (waitKey(1) == 27) {
      break;
    }
  }
  cap.release();
  return 0;
}
