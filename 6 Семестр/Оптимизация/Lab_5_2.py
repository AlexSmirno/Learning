import random
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x1_list = []
x2_list = []
y_list = []
counter = 0
min_val = -1500000

def show(x1_list, x2_list):
    N = int(x1_list.__len__())
    if (N <= 0):
        return

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    x1_array = np.arange(min(x1_list) - 1, max(x1_list) + 1, 0.1)
    x2_array = np.arange(min(x2_list) - 1, max(x2_list) + 1, 0.1)

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

x0_nplist = np.array(0)
x1_nplist = np.array(0)
x2_nplist = np.array(0)

def f(x1, x2):
    return 3*x1**4 - x1*x2 + x2**4 - 7*x1 - 8*x2 + 2

def calc_r(a, n):
    return a * (sqrt(n + 1) - 1 + n) / (n * sqrt(2))

def calc_s(a, n):
    return a * (sqrt(n + 1) - 1) / (n * sqrt(2))

def dist(x1, x2):
    return sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)

def max_f(f):
    f_max = max(f)
    index = f.index(f_max)
    return index

def calc_x_next(x, index, n):
    res = np.array([0,0])
    for i in range(len(x)):
        if (i == index): continue
        res = res + x[i]  
    res *= (2 / (n - 1))
    res -= x[index]
    return res

def calc_centr(x):
    return (x[0][0]+x[1][0]+x[2][0]) / 3, (x[0][1]+x[1][1]+x[2][1]) / 3

def simplexnyi_method(x0, x1, x2, a, n):
    global counter; 
    k = 0
    x = [x0, x1, x2]
    counter += 3
    while (dist(x[0], x[1]) > e or dist(x[1], x[2]) > e or dist(x[2], x[0]) > e):
        f_list = []
        x1_list.append(x[0][0]); x2_list.append(x[0][1])
        x1_list.append(x[1][0]); x2_list.append(x[1][1])
        x1_list.append(x[2][0]); x2_list.append(x[2][1])

        counter += 1
        f_list.append(f(x[0][0], x[0][1]))
        f_list.append(f(x[1][0], x[1][1]))
        f_list.append(f(x[2][0], x[2][1]))

        while(True):
            f_values = f_list
            i = max_f(f_values)
            xn = calc_x_next(x, i, n)
            fn = f(xn[0], xn[1]); counter += 1

            if (f_values[i] > fn): x[i] = xn ; break

            f_values[i] = min_val

            if (f_values[0] == min_val and f_values[1] == min_val and f_values[2] == min_val):
                a /= 2
                x[0] = x[0]
                x[1] = np.array([x[0][0] + calc_r(a, n), x[0][1] + calc_s(a, n)])
                x[2] = np.array([x[0][0] + calc_s(a, n), x[0][1] + calc_r(a, n)])
                break

        k += 1
    point = calc_centr(x)
    return [(point, f(point[0], point[1])), k]

round_num = 3
e = 0.0001
a = 2
n = 3

x0 = np.array([2, 2])
x1 = np.array([x0[0] + calc_r(a, n), x0[1] + calc_s(a, n)])
x2 = np.array([x0[0] + calc_s(a, n), x0[1] + calc_r(a, n)])


result = simplexnyi_method(x0, x1, x2, a, n)
x,y = result[0]
result[0] = ((round(x[0], round_num),round(x[1], round_num)), round(y, round_num))
print(f"start {x0}")
print(f"Simplexnyi method: {result[0]}; count of iteractions = {result[1]}")
print('Count of compute function =', counter)

#show(x1_list, x2_list)
