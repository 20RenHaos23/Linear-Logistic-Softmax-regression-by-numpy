#Linear regression練習

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

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

'''
----------------------------------------------------------------------------------------
'''


def draw_line(plt,w,b,x,linewidth =2):
    m=len(x)
    f = [0]*m
    for i in range(m): 
       f[i] = b+w*x[i]
    plt.plot(x, f, linewidth) 
  
fig, ax = plt.subplots() 
plt.scatter(X, y, marker="x", c="red")
plt.title("Food Truck Dataset", fontsize=16)
plt.xlabel("City Population in 10,000s", fontsize=14)
plt.ylabel("Food Truck Profit in 10,000s", fontsize=14)
plt.axis([4, 25, -5, 25])
w,b = history[-1]
draw_line(plt,w,b,X,6)
plt.show()


def loss(x,y,w,b):
    
    return np.mean((x*w+b-y)**2)/2 #再除2 為根據3-8頁寫的
    '''
    
    #另一種方式
    m = len(y)
    cost = 0   
    for i in range(m):  
        f =  x[i]*w+b
        cost += (f-y[i])**2
    cost /=(2*m) #除2m 為根據3-8頁寫的
    return cost
    '''

#print(loss(X,y,1,-3))

costs = [loss(X,y,w,b) for w,b in history]
plt.axis([0, len(costs), 4, 6])#設置圖形的坐標軸範圍，x軸的範圍是0到len(costs)，y軸的範圍是4到6。
plt.plot(costs)
plt.grid()
plt.show()

def plot_history(x,y,history,figsize=(20,10)):
    w= [ e[0] for e in history]
    b= [ e[1] for e in history]
    #生成 w 和 b 的網格
    xmin,xmax, xstep = min(w)-0.2,max(w)+0.2, .2 # 0.26 1.50 0.2
    ymin, ymax, ystep = min(b)-0.2,max(b)+0.2, .2 # -3.98 0.31 0.2
    ws,bs = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
    
    #計算損失函數
    zs = np.array([loss(x, y, w,b)    for w,b in zip(np.ravel(ws), np.ravel(bs))]) #高
    z = zs.reshape(ws.shape)
    
    #3D繪圖的基本設置
    fig = plt.figure(figsize=figsize)
    #ax = fig.add_subplot(111, projection='3d')
    ax = plt.axes(projection='3d')
    
    ax.set_xlabel('w', labelpad=30, fontsize=24, fontweight='bold')
    ax.set_ylabel('b', labelpad=30, fontsize=24, fontweight='bold')
    ax.set_zlabel('L(w,b)', labelpad=30, fontsize=24, fontweight='bold')
    
    #繪製損失函數的3D表面圖
    ax.plot_surface(ws, bs, z, rstride=1, cstride=1, color='b', alpha=0.2)
    
    #繪製優化過程的起點和終點
    w_sart,b_start,w_end,b_end = history[0][0], history[0][1],history[-1][0], history[-1][1]
    ax.plot([w_sart],[b_start], [loss(x,y,w_sart,b_start)] , markerfacecolor='b', markeredgecolor='b', marker='o', markersize=7)
    ax.plot([w_end],[b_end], [loss(x,y,w_end,b_end)] , markerfacecolor='r', markeredgecolor='r', marker='o', markersize=7)
    
    #描繪梯度下降的軌跡
    z2 =  [loss(x,y,w,b) for w,b in history]
    ax.plot(w, b, z2  , markerfacecolor='r', markeredgecolor='r', marker='.', markersize=2) #點之間會自動用線連接起來
    ax.plot(w, b,  0 , markerfacecolor='r', markeredgecolor='r', marker='.', markersize=2) #畫成平面的樣子 z軸皆為0
    
    fig.suptitle("L(w,b)", fontsize=24, fontweight='bold')
    plt.show()
    return ws,bs,z
    
ws,bs,z = plot_history(X,y,history)


#繪製等高線圖:
plt.figure()
plt.contour(bs,ws,z,levels=np.logspace(-5, 5, 100), norm=LogNorm(), cmap=plt.cm.jet) 

w= [ e[0] for e in history]
b= [ e[1] for e in history]
plt.plot(b,w)
plt.xlabel("b")
plt.ylabel("w")
title = str.format("iteration={0}, alpha={1}, b={2:.3f}, w={3:.3f}", iterations, alpha, b[-1], w[-1])
plt.title(title)
plt.show()

#使用不同learning rate，並畫出圖
plt.figure()
num_iters = 1200
learning_rates = [0.01, 0.015, 0.02]
for lr in learning_rates:
    w,b=0,0 
    history = gradient_descent(X, y,w, b,lr, num_iters)
    cost_history = [loss(X,y,w,b) for w,b in history]
    plt.plot(cost_history, linewidth=2)
plt.title("Gradient descent with different learning rates", fontsize=16)
plt.xlabel("number of iterations", fontsize=14)
plt.ylabel("cost", fontsize=14)
plt.legend(list(map(str, learning_rates)))
plt.axis([0, num_iters, 4, 6])
plt.grid()
plt.show()


#梯度驗證
df_approx = lambda x,y,w,b,eps: ( (loss(x,y,w+eps,b)-loss(x,y,w-eps,b) )/(2*eps),  (loss(x,y,w,b+eps)-loss(x,y,w,b-eps) )/(2*eps) )
#在任意一个点如 (𝑤,𝑏)=(1.0,−2.0) 比较分析和数值梯度。
w = 1.0
b = -2.
eps = 1e-8
dw = np.mean((w*X+b-y)*X)
db = np.mean((w*X+b-y))                                   
grad = np.array([dw,db])
grad_approx = np.array(df_approx(X,y,w,b,eps))
print()
print('梯度驗證')
print(grad)
print(grad_approx)
print(abs(grad-grad_approx))



#用求得的w b求X种样本的预测值
m=len(X)
predictions = [0]*m   
for i in range(m): 
    predictions[i] =  X[i]*1.1822480052540145 + -3.7884192615511796

plt.scatter(X, y, marker="x", c="red")
plt.scatter(X, predictions, marker="o", c="blue") 
#plt.plot(X, predictions, linewidth=2)  # plot the hypothesis on top of the training data
plt.show()