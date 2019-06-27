#!/usr/bin/python3
import numpy as np
import cv2

class Projeto():
  def calibrate(self, board=(6,6)):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    points = np.zeros((board[0]*board[1],3), np.float32)
    points[:,:2] = np.mgrid[0:board[0],0:board[1]].T.reshape(-1,2)
    
    goal = 10 # número de imagens pra calibração
    objpoints = [None] * goal # pontos no espaço 3D
    imgpoints = [None] * goal # pontos no plano de imagem
    vc = cv2.VideoCapture(0)
    n = 0
    while n < goal:
      ret, frame = vc.read()
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      ret, corners = cv2.findChessboardCorners(gray, board, None)
      if ret == True:
        objpoints[n] = points
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints[n] = corners
        frame = cv2.drawChessboardCorners(frame, board, corners, ret)
        n += 1
        print('Imagem capturada {}/{}'.format(n, goal))
      cv2.imshow('img', gray)
      cv2.waitKey(1)
    vc.release()
    cv2.destroyAllWindows()
    shape = gray.shape
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, gray.shape, 1, gray.shape)
    self.cameramatrix = newcameramatrix
    self.mtx = mtx
		self.dist = dist
		self.roi = roi
  def undistort(self, img):
    dst = cv2.undistort(img, self.mtx, self.dist, None, self.cameramatrix)
		x,y,w,h = self.roi
		dst = dst[y:y+h, x:x+w]
		return dst

p = Projeto()
p.calibrate()
