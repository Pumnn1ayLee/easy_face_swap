import numpy as np
import cv2
import dlib
font = cv2.FONT_HERSHEY_SIMPLEX
# dlib预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("E://Download/shape_predictor_68_face_landmarks.dat.dat")
# cv2读取图像
img = cv2.imread('D://pythonProject5/face.jpg')
chg = cv2.imread('D://pythonProject5/face.jpg')
# 获取人脸图像rects
rects = detector(img)
#获取 68 个点坐标保存在landmarks中，
landmarks1 = np.matrix([[p.x, p.y] for p in predictor(chg, rects[0]).parts()])
x17_0=landmarks1[17,0]
x26_0=landmarks1[26,0]
y18_0=landmarks1[18,1]
y40_0=landmarks1[28,1]

x31_0=landmarks1[49,0]
x35_0=landmarks1[54,0]
y29_0=landmarks1[28,1]
y32_0=landmarks1[57,1]


landmarks2 = np.matrix([[p.x, p.y] for p in predictor(chg, rects[1]).parts()])
x17_1=landmarks2[17,0]
x26_1=landmarks2[26,0]
y18_1=landmarks2[18,1]
y40_1=landmarks2[28,1]

x31_1=landmarks2[49,0]
x35_1=landmarks2[54,0]
y29_1=landmarks2[28,1]
y32_1=landmarks2[57,1]

w1=x26_0-x17_0+5
w2=x26_1-x17_1+5
l1=y40_0-y18_0+5
l2=y40_1-y18_1+5

j1=x35_0-x31_0+12
j2=x35_1-x31_1+12
k1=y32_0-y29_0+10
k2=y32_1-y29_1+10

eye1=img[y18_0-7:y18_0-3+max(l1,l2),x17_0-7:x17_0-3+max(w1,w2)]
eye2=img[y18_1-7:y18_1-3+max(l1,l2),x17_1-7:x17_1-3+max(w1,w2)]
chg[y18_1-7:y18_1-3+max(l1,l2),x17_1-7:x17_1-3+max(w1,w2)]=eye1
chg[y18_0-7:y18_0-3+max(l1,l2),x17_0-7:x17_0-3+max(w1,w2)]=eye2

noseandmouth1=img[y29_0-5:y29_0-2+max(k1,k2),x31_0-17:x31_0-3+max(j1,j2)]
noseandmouth2=img[y29_1-5:y29_1-2+max(k1,k2),x31_1-17:x31_1-3+max(j1,j2)]
chg[y29_0-5:y29_0-2+max(k1,k2),x31_0-17:x31_0-3+max(j1,j2)]=noseandmouth2
chg[y29_1-5:y29_1-2+max(k1,k2),x31_1-17:x31_1-3+max(j1,j2)]=noseandmouth1

cv2.imshow('img',chg)
cv2.waitKey(0)