import random
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x1_list = []
x2_list = []
y_list = []
counter = 0

def drawFunc(minX, minY, maxX, maxY):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    x1_array = np.arange(minX, maxX, 0.1)
    x2_array = np.arange(minY, maxY, 0.1)
    x1_array, x2_array = np.meshgrid(x1_array, x2_array)
    R = f(x1_array, x2_array)

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1,x2)')   
    ax.plot_surface(x1_array, x2_array, R, color='b', alpha=0.5) 
    plt.show()

def show(x1_list, x2_list):
    N = int(x1_list.__len__())
    if (N <= 0):
        return

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    x1_array = np.arange(min(x1_list) - 0.1, max(x1_list) + 0.1, 0.1)
    x2_array = np.arange(min(x2_list) - 0.1, max(x2_list) + 0.1, 0.1)

    x1_array, x2_array = np.meshgrid(x1_array, x2_array)
    R = f(x1_array, x2_array)


    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1,x2)')   
    ax.plot_surface(x1_array, x2_array, R, color='b', alpha=0.5) 
    
    x1_list2 = []
    x2_list2 = []
    f_list = []

    ax.scatter(x1_list[0], x2_list[0], f(x1_list[0], x2_list[0]), c='black')
    x1_list2.append(x1_list[0])
    x2_list2.append(x2_list[0])
    f_list.append(f(x1_list[0], x2_list[0]))

    for n in range(1, N - 1):
        ax.scatter(x1_list[n], x2_list[n], f(x1_list[n], x2_list[n]), c='red')
        x1_list2.append(x1_list[n])
        x2_list2.append(x2_list[n])
        f_list.append(f(x1_list[n], x2_list[n]))

    ax.scatter(x1_list[N - 1], x2_list[N - 1], f(x1_list[N - 1], x2_list[N - 1]), c='green')
    x1_list2.append(x1_list[N - 1])
    x2_list2.append(x2_list[N - 1])
    f_list.append(f(x1_list[N - 1], x2_list[n]))

    ax.plot(x1_list2, x2_list2, f_list, color="black")

    plt.show()

def f(x1, x2):
    return 3*x1**4 - x1*x2 + x2**4 - 7*x1 - 8*x2 + 2

def find_s(x1, x2, a, h):
    global counter; counter += 1
    return (f(x1 + a*h[0], x2 + a*h[1]) - f(x1, x2)) / a

def rand(n):
    return random.randint(0, n)


def random_gradient_decent(x1, x2, e, a):
    k = 0
    while True:
        ej = [0, 0]
        j = rand(2)
        ej[j-1] = 1

        sk = find_s(x1, x2, a, ej)
        x1_next = x1 - a*sk*ej[0]
        x2_next = x2 - a*sk*ej[1]

        x1_list.append(x1); x2_list.append(x2)

        if (sqrt(abs(x1_next - x1)**2 + abs(x2_next - x2)**2) <= e):
            return [(round(x1_next, round_num), 
                     round(x2_next, round_num), 
                     round(f(x1_next, x2_next), round_num)), 
                    k]
        x1 = x1_next
        x2 = x2_next
        k += 1


round_num = 4
x1 = 0
x2 = 0
e = 0.00001
a = 0.01

result = random_gradient_decent(x1, x2, e, a)
print(f"Random gradient decent: {result[0]}; count of iteractions = {result[1]}")
print('Count of compute function =', counter + 1)


#show(x1_list, x2_list)
#drawFunc(-10, -10, 10, 10)


