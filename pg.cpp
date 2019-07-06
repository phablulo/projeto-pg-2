#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;
#define wname "image"

struct CalibrateResult {
  Mat intrinsic;
  Mat distCoeffs;
};

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
    imshow(wname, image);
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
    
    imshow(wname, image);
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
    imshow(wname, image);
    if (waitKey(1) == 27) {
      break;
    }
  }
  return 0;
}

int main(int argc, char* argv[]) {
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
        return 1;
      }
      track(cap, image, calibration);
    }
    else {
      status = -1;
    }
    status += 1;
  }
  cap.release();
  return 0;
}
