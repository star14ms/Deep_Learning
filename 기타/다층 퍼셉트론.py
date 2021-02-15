# 2진수를 10진수로 변환

import numpy as np

# 2진수 -> 10진수 변환 함수
def Convert_binary_num_to_decimal(binary_num):

    x = [] # 각 자릿수 분리
    for i in binary_num:
        if not (i=='0' or i=='1'):
            print("Error: 이진수가 아님")
            return False
        x.append(int(i))
    x = np.array(x)
    
    length = len(binary_num) # 각 자릿수의 영향력(가중치) 정의
    w = np.zeros(length)
    for i in range(length):
        w[i] = 2**i

    return np.sum(x*w) # 십진수로 변환

# 10진수 -> 2진수 변환 함수
def Convert_decimal_num_to_binary(decimal_num):

    for i in decimal_num:
        try:
            if not (0<=int(i)<=9):
                raise
        except:
            print("Error: 십진수가 아님")
            return False
    
    num = int(decimal_num)

    n = 1
    while n <= num:
        n *= 2

    info = ""
    while n != 1:
        n /= 2
        if num >= n:
            info += '1'
            num -= n
        else:
            info += '0'
    
    return int(info)

input = input()
# print(Convert_binary_num_to_decimal(input))
print(Convert_decimal_num_to_binary(input))