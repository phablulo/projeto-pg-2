#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

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
    cvtColor(image, imageGray, CV_BGR2GRAY);
    bool found = findChessboardCorners(image, boardSize, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
    if (found) {
      cout << "Found! " << ++success*100/numBoards << '%' << endl;
      cornerSubPix(imageGray, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
      drawChessboardCorners(imageGray, boardSize, corners, found);
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
  while (true) {
    cap >> image;
    undistort(image, undistorted, intrinsic, distCoeffs);
    imshow("image", undistorted);
    if (waitKey(1) == 27) {
      break;
    }
  }
  cap.release();
  return 0;
}
