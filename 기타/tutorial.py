import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
# print(dir(np))

# 1.헬로 파이썬 #

# numpy
# A = np.array([[1, 2], [3, 4]])
# print(A.shape, A.dtype, A.flatten())
# for row in A:
    # print(row)
# print(A[np.array([0,1])])

# # sin, cos 그래프
# x = np.arange(0, 2*np.pi, 0.1) # np.arange(시작, 끝, 간격)
# y1 = np.sin(x) # .sin
# y2 = np.cos(x) # .cos

# plt.plot(x, y1, label="sin") # plt.plot
# plt.plot(x, y2, linestyle="--", label="cos") # label=, linestyle=

# plt.xlabel("x"k) # .xlabel
# plt.ylabel("y") # .ylabel
# plt.title('sin & cos') # .title
# plt.legend() # .legend (그래프 이름 표시)
# plt.show() # .show

# # 이미지 표시k
# img = imread(r"C:\Users\danal\OneDrive\바탕 화면\programing\python\crawler\search naver image\img\사과\사과1.jpg") # .imread
# plt.imshow(img) # .imshow
# plt.show


# 2.퍼셉트론 #

# def AND(x1, x2):
#     w1, w2, theta = 0.5, 0.5, 0.9
#     tmp = x1*w1 + x2*w2
#     if tmp <= theta:
#         return 0
#     elif tmp > theta:
#         return 1

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.9

    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.9

    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.4

    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def XOR(x1, x2):
    return AND(NAND(x1, x2), OR(x1, x2))

# print(AND(0, 0), AND(0, 1), AND(1, 0), AND(1, 1))
# print(NAND(0, 0), NAND(0, 1), NAND(1, 0), NAND(1, 1))
# print(OR(0, 0), OR(0, 1), OR(1, 0), OR(1, 1))
# print(XOR(0, 0), XOR(0, 1), XOR(1, 0), XOR(1, 1))

# 3.신경망 #

# 3-2 활성화 함수: 입력신호의 총합을 출력신호로 변환하는 함수
def step(x):
    return np.array(x > 0, dtype=np.int) # y = x>0, y.astype(np.int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)

# y = step(x)
y = sigmoid(x)
# y = ReLU(x)

# plt.plot(x, y)
# plt.ylim(-0.1, 1.1)
# plt.ylim(-1, 5.5)
# plt.ylim(-1.1, 1.1)
# plt.show()


# 3-3 다차원 배열의 계산
A = np.array([[1,2],[3,4],[5,6]])
# print(np.ndim(A))
# print(A.shape)
# print(A.shape[0])

A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])
# print(np.dot(A, B))


# 3-5 출력층 설계하기 (활성화 함수)

# 항등 함수 (회귀)
def identity(x):
    return x
    
# 소프트맥스 함수 (분류) (2클래스 분류: 시그모이드)
def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c)
    y = exp_x / np.sum(exp_x)
    return y

x = np.array([0.1, 0.5, 1.0])
y = softmax(x)
# print(y, sum(y))

# 4.2 손실 함수

# 오차제곱합
def sum_squares_error(y, t):
    return 0.5 * np.sum((y-t)**2)

# 교차 엔트로피 오차
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y+delta))

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]

t, y1, y2 = np.array(t), np.array(y1), np.array(y2)

# print(sum_squares_error(y1, t), sum_squares_error(y2, t))
# print(cross_entroopy_error(y1, t), cross_entroopy_error(y2, t))

# 4-3 수치 미분

# 중심 차분 (중앙 차분)
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

# 수치 미분
def function_1(x):
    return 0.01*x**2 + 0.1*x

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()

print(numerical_diff(function_1, 5), numerical_diff(function_1, 10))

# 편미분
def function_2(x):
    return x[0]**2 + x[1]**2

def function_tmp1(x0):
    return x0*x0 + 4*4

def function_tmp2(x1):
    return 3*3 + x1*x1

print(numerical_diff(function_tmp1, 3), numerical_diff(function_tmp2, 4))

# 4-4 기울기 (모든 변수의 편미분을 벡터로 정리한 것)
def numerical_gradient(f, x):
    h = 1
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]  

        x[idx] = tmp_val + h
        fxh1 = f(x)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] =  tmp_val

    return grad

print(numerical_gradient(function_2, np.array([3, 4])))
print(numerical_gradient(function_2, np.array([0, 2])))
print(numerical_gradient(function_2, np.array([3, 0])))

# 오차 역전파법

# 5-4 단순한 계층 구현하기
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return (x * y)

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy

class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        return x + y

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class sigmoid:
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

def Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss == cross_entropy_error(self.y, self.t)
        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx
