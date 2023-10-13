#Linear regression練習

import matplotlib.pyplot as plt
import numpy as np

#讀取資料
x , y = [] ,[]
with open('food_truck_data.txt') as A:
    for eachline in A:
        s = eachline.split(',')
        x.append(float(s[0]))
        y.append(float(s[1]))  

    
#將讀取的資料畫出來
fig, ax = plt.subplots()   
ax.scatter(x, y, marker="x", c="red")
plt.title("Food Truck Dataset", fontsize=16)
plt.xlabel("City Population in 10,000s", fontsize=14)
plt.ylabel("Food Truck Profit in 10,000s", fontsize=14)
plt.axis([4, 25, -5, 25])#設置圖形的坐標軸範圍，x軸的範圍是4到25，y軸的範圍是-5到25。
plt.show()



#使用正規方程法求解線性回歸模型的參數
data = np.loadtxt('food_truck_data.txt', delimiter=",") # data是m*2矩阵
train_x = data[:, 0]    # 城市人口，  m*1矩阵 
train_y = data[:, 1]    # 餐车利润，  m*1矩阵

#X為m*2矩陣
X = np.ones(shape=(len(train_x), 2))
X[:, 1] = train_x 
#y為m*1矩陣
y = train_y

#XT為X矩陣作轉置
XT = X.transpose()

#XTy為XT跟y做矩陣相乘
XTy = XT @ y
#XTy = np.dot(XT, y) #也可以用這種 矩陣相乘
#XTy = XT.dot(y) #也可以用這種 矩陣相乘

#w為XT跟X做矩陣相乘後做反矩陣再與XTy做矩陣相乘
w = np.linalg.inv(XT@X) @ XTy
print('使用正規方程法求解線性回歸模型的參數')
print('w : {},b : {}'.format(w[1],w[0]))

'''
----------------------------------------------------------------------------------------
'''

#求解線性迴歸的梯度下降法演算法的程式碼
X = train_x
w,b = 0.,0.
dw = np.mean((w*X+b-y)*X)
db = np.mean((w*X+b-y))


def gradient_descent(x,y,w,b,alpha=0.01, iterations = 100,epsilon = 1e-9):    
    history=[]
    for i in range(iterations):
        dw = np.mean((w*x+b-y)*x)
        db = np.mean((w*x+b-y))       
        if abs(dw) < epsilon and abs(db) < epsilon:
           break;
     
        #更新w: w = w - alpha * gradient
        w -= alpha*dw 
        b -= alpha*db 
        history.append([w,b])  
       
    return history

alpha = 0.02
iterations=1000
history = gradient_descent(X,y,w,b,alpha,iterations)
print()
print('使用梯度下降法求解線性回歸模型的參數')
print('w : {},b : {}'.format(history[-1][0],history[-1][1]))
