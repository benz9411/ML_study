import cv2
import numpy as np
from matplotlib import pyplot as plt

#각 데이터의 위치 : 25 x 2 크기에 0 ~100
trainData = np.random.randint(0,100,(25,2)).astype(np.float32)
#각 데이터는 0 or 1
response = np.random.randint(0,2 ,(25,1)).astype(np.float32)

#값이 0인 데이터를 각각 (x,y) 위치에 빨간색으로 칠합니다.

#값이 0인 데이터를 각각 (x,y) 위치에 빨간색으로 칠합니다.
red = trainData[response.ravel() == 0]
plt.scatter(red[:,0],red[:,1],80, 'r', '^') # x좌표 y좌표 r은 빨간색 '^' 세모라는 뜻
#깂이 1인 데이터를 가각 (x,y) 위치에 파란색으로 칠합니다.
blue = trainData[response.ravel() == 1]
plt.scatter(blue[:,0], blue[:,1],80,'b','s') #이건 네모

newcomer = np.random.randint(0, 100, (1, 2)).astype(np.float32)
plt.scatter(newcomer[:,0], newcomer[:,1],80,'g','o')


knn = cv2.ml.KNearest_create() #이 함수를 이용해서 객체 초기화
# opencv에 내장된 knn 알고리즘 클래스를 제공함
knn.train(trainData, cv2.ml.ROW_SAMPLE, response) #train 함수 
ret, result, neighbours, dist = knn.findNearest(newcomer,3)

print("result :" , result)
print("neighbours :", neighbours)
print("distance:", dist)

plt.show()