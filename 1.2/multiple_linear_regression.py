#参考https://www.cnblogs.com/magic-girl/p/mutivariable-linear-regression.html
from numpy import genfromtxt
import numpy as np
from numpy.linalg import pinv

dataPath = r"Delivery.csv"
# 加上r就会将后面当做一个完整的字符串，而不会认为里面有什么转义之类的。
deliveryData = genfromtxt(dataPath, delimiter=',')

print("data")
print(deliveryData)

x = deliveryData[:, :-1]
y = deliveryData[:, -1]

print("x: ")
print(x)
print("y: ")
print(y)
#一维numpy数组转置方法
y = y.reshape(len(y),-1)
print(y)

# 标准方程法（Normal Equation）
# theta = pinv(x.T.dot(x)).dot(x.T).dot(y)

# 梯度下降法（Gradient Descent）
#δJδθ=1/mXT(Xθ−y)
def partial_derivative(X, theta, y):
    print(X.shape, theta.shape, y.shape)
    print(X.T.dot(X.dot(theta) - y))
    derivative = X.T.dot(X.dot(theta) - y) / X.shape[0]
    print(derivative)
    return derivative


def gradient_descent(X, y, alpha=0.0001):
    print(X.shape[1])
    theta = np.ones(shape=X.shape[1], dtype=float)
    theta = theta.reshape(len(theta), -1)
    partial_derivative_of_J = partial_derivative(X, theta, y)
    while any(abs(partial_derivative_of_J) > 1e-5):
        theta = theta - alpha * partial_derivative_of_J
        print("theta.shape", theta.shape)
        partial_derivative_of_J = partial_derivative(X, theta, y)
    return theta


theta = gradient_descent(x, y)
print(theta)
# print(theta.shape)

# 预测一个例子
xPred = np.array([103, 7])
# print(xPred.shape)
print(xPred)
# hθ(x)=θT∙x
yPred = theta.T.dot(xPred)
print("predicted y:")
print(yPred)
#[ 11.32750806]
