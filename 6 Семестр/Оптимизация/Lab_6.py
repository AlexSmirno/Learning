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

    x1_array = np.arange(0, 5, 0.1)
    x2_array = np.arange(0, 5, 0.1)
    x1_array, x2_array = np.meshgrid(x1_array, x2_array)
    R = f(x1_array, x2_array)

    #drawBoder(ax, x1_array, g1_1)
    #drawBoder(ax, x1_array, g2_1)
    #drawBoder(ax, x1_array, g3_1)
    #drawBoder(ax, x1_array, g4_1)
    
    ax.plot_surface(x1_array, x2_array, R, color='b', alpha=0.5)
    
    plt.show()

def drawBoder(ax, x1, g):
    zs = np.arange(0, 80, 35)

    X, Z = np.meshgrid(x1, zs)
    Y = g(X)
    ax.plot_surface(X, Y, Z, alpha=0.3)

def show(x1_list, x2_list):
    N = int(x1_list.__len__())
    if (N <= 0):
        return

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    x1_array = np.arange(min(x1_list) - 0.1, max(x1_list) + 0.1, 0.1)
    x2_array = np.arange(min(x2_list) - 0.1, max(x2_list) + 0.1, 0.1)

    x1_array, x2_array = np.meshgrid(x1_array, x2_array)
    R = f(x1_array, x2_array)
    
    drawBoder(ax, x1_array, g1_1)
    drawBoder(ax, x1_array, g2_1)
    drawBoder(ax, x1_array, g3_1)
    drawBoder(ax, x1_array, g4_1)


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


def g1_t(x1, x2):
    return -3*x1 - 2*x2 + 6 <= 0

def g2_t(x1, x2):
    return -x1 + x2 - 3 <= 0

def g3_t(x1, x2):
    return x1 + x2 - 7 <= 0

def g4_t(x1, x2):
    return 2*x1 - 3*x2 - 4 <= 0

def F(x1, x2, r):
    #sum = 1/g1(x1, x2) + 1/g2(x1, x2) + 1/g3(x1, x2) + 1/g4(x1, x2)
    #- r * sum
    return f(x1,x2) + P(x1, x2, r)

def F2(x1, x2, r):
    sum = log(-g1(x1, x2)) + log(-g2(x1, x2)) + log(-g3(x1, x2)) + log(-g4(x1, x2))

    return f(x1,x2) - r * sum

def P(x1, x2, r):
    sum = 1/g1(x1, x2) + 1/g2(x1, x2) + 1/g3(x1, x2) + 1/g4(x1, x2)

    return -r*sum

def P1(x1, x2, r):
    sum = log(-g1(x1, x2)) + log(-g2(x1, x2)) + log(-g3(x1, x2)) + log(-g4(x1, x2))
    
    return -r*sum

def find_s(x1, x2, a, h, F_x, r):
    global counter; counter += 1
    return (F_x(x1 + a*h[0], x2 + a*h[1], r) - F_x(x1, x2, r)) / a

def rand(n):
    return random.randint(0, n)

def random_gradient_decent(x1, x2, e, a, F_x, r):
    k = 0
    while True:
        ej = [0, 0]
        j = rand(2)
        ej[j-1] = 1

        sk = find_s(x1, x2, a, ej, F_x, r)
        x1_next = x1 - a*sk*ej[0]
        x2_next = x2 - a*sk*ej[1]

        x1_list.append(x1); x2_list.append(x2)

        if (not (g1_t(x1_next, x2_next) and g2_t(x1_next, x2_next) and g3_t(x1_next, x2_next) and g4_t(x1_next, x2_next))):
            return (round(x1, round_num), 
                     round(x2, round_num))

        if (sqrt(abs(x1_next - x1)**2 + abs(x2_next - x2)**2) <= e):
            return (round(x1_next, round_num),
                     round(x2_next, round_num))
        
        x1 = x1_next
        x2 = x2_next
        k += 1


def barrier_function_method(x1, x2, r, C, e, a, k):
    min_x1, min_x2 = random_gradient_decent(x1, x2, e, a, F, r)
    #print("x1 =", min_x1, "x2 =", min_x2)
    fine = P(min_x1, min_x2, r)
    #print("fine =", fine)
    if (abs(fine) <= e):
            return [(round(min_x1, round_num), 
                     round(min_x2, round_num), 
                     round(f(min_x1, min_x2), round_num)), 
                    k]
    k += 1
    r = r/C
    return barrier_function_method(min_x1, min_x2, r, C, e, a, k)


round_num = 4
x1 = 2
x2 = 2
e = 0.001
a = 0.01
r = 1
c = 16
k = 0

result = barrier_function_method(x1, x2, r, c, e, a, k)
print(f"Barrier function method: {result[0]}; count of iteractions = {result[1]}")
print('Count of compute function =', counter + 1)



#show(x1_list, x2_list)
drawFunc(-5, -5, 15, 15)