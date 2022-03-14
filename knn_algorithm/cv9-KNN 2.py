import cv2
import numpy as np

# digits 라는 사진은 20x20으로 5천개 사진이 있고 그걸 우리가 분할하면 5000개의 데이터를 얻는다
img = cv2.imread('images/digits.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#세로로 50줄 , 가로로 100줄로 사진을 나눕니다.
cells = [np.hsplit(row, 100) for row in np.vsplit(gray,50)]
x = np.array(cells)
print(x.shape) # 50 100 20 20

# 각 (20 x 20 ) 크기의 사진을 한 줄 (1x400)으로 바꿉니다.
# 20 x 20 을 한 줄로 길게 늘리는 것 머신러닝 때는 데이터를 이렇게 한 줄 즉 벡터 형태로 넣는다
train = x[:,:].reshape(-1,400).astype(np.float32)

k=np.arange(10)
# train_labels는 이게 0~9인지 확인한 레이블이다.
train_labels = np.repeat(k,500)[:,np.newaxis]

#이건 학습 데이터로 쓰겠다는 말 npz
np.savez("trained.npz", train=train, train_labels=train_labels)

print(x[0,5].shape)

cv2.imshow('image',x[0,5])
cv2.waitKey(0)

cv2.imwrite('test0.png',x[0,0])
cv2.imwrite('test1.png',x[5,0])
cv2.imwrite('test2.png',x[10,0])
cv2.imwrite('test3.png',x[15,0])
cv2.imwrite('test4.png',x[20,0])
cv2.imwrite('test5.png',x[25,0])
cv2.imwrite('test6.png',x[30,0])
cv2.imwrite('test7.png',x[35,0])
cv2.imwrite('test8.png',x[40,0])
cv2.imwrite('test9.png',x[45,0])
