import random
import numpy as np
from math import sqrt, log
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x1_list = []
x2_list = []
y_list = []
counter = 0

def drawFunc(minX, minY, maxX, maxY):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1,x2)')

    x1_array = np.arange(minX, maxX, 0.1)
    x2_array = np.arange(minY, maxY, 0.1)
    x1_array, x2_array = np.meshgrid(x1_array, x2_array)
    R = f(x1_array, x2_array)
    
    ax.plot_surface(x1_array, x2_array, R, color='b', alpha=0.5)
    
    plt.show()

def drawBoder(ax, x1, g, z_min, z_max):
    zs = np.arange(0, 300, 100)

    X, Z = np.meshgrid(x1, zs)
    Y = g(X)

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z)

def show(x1_list, x2_list):
    N = int(x1_list.__len__())
    if (N <= 0):
        return

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    x1_array = []
    x2_array = []
    #x1_array = np.arange(min(x1_list) - 0.1, max(x1_list) + 0.1, 0.1)
    #x2_array = np.arange(min(x2_list) - 0.1, max(x2_list) + 0.1, 0.1)
    nums = np.arange(0, 5, 0.1)
    for i in range(len(nums)):
        for j in range(len(nums)):
            if(barier(nums[i], nums[j])):
                x1_array.append(nums[i])
                x2_array.append(nums[j])

    x1_array = np.array(x1_array)
    x2_array = np.array(x2_array)
    x1_array, x2_array = np.meshgrid(x1_array, x2_array)
    R = f(x1_array, x2_array)
    
    #drawBoder(ax, x1_array, g1_1, R.min(), R.max())
    #drawBoder(ax, x1_array, g2_1, R.min(), R.max())
    #drawBoder(ax, x1_array, g3_1, R.min(), R.max())
    #drawBoder(ax, x1_array, g4_1, R.min(), R.max())


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
    
def f_1(x1, x2):
    if (g1_1(x1,x2) and g2_1(x1,x2) and g3_1(x1,x2) and g4_1(x1,x2)):
        return (x1-6)**2 +(x2-7)**2
    return 0

def g1_1(x1):
    return (-3*x1 + 6) / 2

def g2_1(x1):
    return (-x1 - 3) / (-1)

def g3_1(x1):
    return (x1 - 7) / (-1)

def g4_1(x1):
    return (2*x1 - 4) / 3


def f(x1, x2):
    return (x1-6)**2 +(x2-7)**2

def g1(x1, x2):
    return -3*x1 - 2*x2 + 6 #<= 0

def g2(x1, x2):
    return -x1 + x2 - 3 #<= 0

def g3(x1, x2):
    return x1 + x2 - 7 #<= 0

def g4(x1, x2):
    return 2*x1 - 3*x2 - 4 #<= 0


def g1_t(x, y):
    return -3*x - 2*y + 6 <= 0

def g2_t(x, y):
    return -x + y - 3 <= 0

def g3_t(x, y):
    return x + y - 7 <= 0

def g4_t(x, y):
    return 2*x - 3*y - 4 <= 0

def F(x1, x2, r):
    #sum = 1/g1(x1, x2) + 1/g2(x1, x2) + 1/g3(x1, x2) + 1/g4(x1, x2)
    #- r * sum
    return f(x1,x2) + P(x1, x2, r)

def F2(x1, x2, r):
    #print("x1 =", x1)
    #print("x2 =", x2)
    #print("gi =", g3(x1, x2))
    #print("log =", log(-g3(x1, x2)))
    
    sum = log(-g1(x1, x2)) + log(-g2(x1, x2)) + log(-g3(x1, x2)) + log(-g4(x1, x2))

    return f(x1,x2) - r * sum

def P(x1, x2, r):
    sum = 1/g1(x1, x2) + 1/g2(x1, x2) + 1/g3(x1, x2) + 1/g4(x1, x2)

    return -r*sum

def P2(x1, x2, r):
    sum = log(-g1(x1, x2)) + log(-g2(x1, x2)) + log(-g3(x1, x2)) + log(-g4(x1, x2))
    
    return -r*sum

min_val = -1500000
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

def barier(x1, x2):
    return not (g1_t(x1, x2) and g2_t(x1, x2) and g3_t(x1, x2) and g4_t(x1, x2))

def simplexnyi_method(x0, e, a, n, r):
    global counter
    x1 = np.array([x0[0] + calc_r(a, n), x0[1] + calc_s(a, n)])
    x2 = np.array([x0[0] + calc_s(a, n), x0[1] + calc_r(a, n)])

    if (barier(x0[0], x0[1])): return;

    while (barier(x1[0], x1[1]) or barier(x2[0], x2[1])):
        a /= 2
        x1 = np.array([x0[0] + calc_r(a, n), x0[1] + calc_s(a, n)])
        x2 = np.array([x0[0] + calc_s(a, n), x0[1] + calc_r(a, n)])

    x = [x0, x1, x2]

    counter += 3
    while (dist(x[0], x[1]) > e or dist(x[1], x[2]) > e or dist(x[2], x[0]) > e):
        #print("center =", calc_centr(x), "f =", f(calc_centr(x)[0], calc_centr(x)[1]))
        if (barier(x[0][0], x[0][1]) or barier(x[1][0], x[1][1]) or barier(x[2][0], x[2][1])):
            return (center[0], center[1], a)
        
        center = calc_centr(x)
        f_list = []     
        x1_list.append(center[0]); x2_list.append(center[1])

        counter += 1
        

        f_list.append(F2(x[0][0], x[0][1], r))
        f_list.append(F2(x[1][0], x[1][1], r))
        f_list.append(F2(x[2][0], x[2][1], r))

        counter += 1
        while(True):
            f_values = f_list
            i = max_f(f_values)
            xn = calc_x_next(x, i, n)
            if (not barier(xn[0], xn[1])):
                fn = F2(xn[0], xn[1], r); counter += 1
                #x_new = x.copy()
                #x_new[i] = xn
                #x_c = calc_centr(x_new)
                if (f_values[i] > fn): x[i] = xn ; break

            f_values[i] = min_val
            if (f_values[0] == min_val and f_values[1] == min_val and f_values[2] == min_val):
                a /= 2
                x[0] = x[0]
                x[1] = np.array([x[0][0] + calc_r(a, n), x[0][1] + calc_s(a, n)])
                x[2] = np.array([x[0][0] + calc_s(a, n), x[0][1] + calc_r(a, n)])
                break
        cur_center = calc_centr(x)
        #print(center)
        if barier(cur_center[0], cur_center[1]):
            return (center[0], center[1], a)

    point = calc_centr(x)
    return (point[0], point[1], a)

def barrier_function_method(x1, x2, r, C, e, a, n, k):
    global counter
    counter += 1

    min_x1, min_x2, a = simplexnyi_method([x1, x2], e, a, n, r)
    fine = P2(min_x1, min_x2, r)

    if (abs(fine) <= e):
            return [(round(min_x1, round_num), 
                     round(min_x2, round_num), 
                     round(f(min_x1, min_x2), round_num)), 
                    k]
    k += 1
    r = r/C
    return barrier_function_method(min_x1, min_x2, r, C, e, a, n, k)

round_num = 3
x1 = 2
x2 = 2
e = 0.001
#a = 0.001
a = 1; n = 3
r = 1
c = 14
k = 0

result = barrier_function_method(x1, x2, r, c, e, a, n, k)
print(f"Barrier function method: {result[0]}; count of iteractions = {result[1]}")
print('Count of compute function =', counter + 1)


show(x1_list, x2_list)
drawFunc(-5, -5, 15, 15)


