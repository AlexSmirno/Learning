import random
import numpy as np
from math import sqrt, log
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x1_list = []
x2_list = []
y_list = []
counter = 0

def drawFunc(minX, minY, maxX, maxY, ax = None):
    #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #ax.set_xlabel('x1')
    #ax.set_ylabel('x2')
    #ax.set_zlabel('f(x1,x2)')

    x1_array = np.arange(minX, maxX, 0.1)
    x2_array = np.arange(minY, maxY, 0.1)
    x1_array, x2_array = fill_arrays(x1_array, x2_array)
    R = fill_z(x1_array, x2_array)
    
    x1_array = np.arange(minX, maxX, 0.1)
    x2_array = np.arange(minY, maxY, 0.1)
    x1_array, x2_array = np.meshgrid(x1_array, x2_array)
    #R = f(x1_array, x2_array)

    #drawBoder(ax, x1_array, g1_1)
    #drawBoder(ax, x1_array, g2_1)
    #drawBoder(ax, x1_array, g3_1)
    #drawBoder(ax, x1_array, g4_1)

    #print(R)
    ax.plot_surface(x1_array, x2_array, R, alpha = 0.6)
    #plt.show()

def fill_arrays(x, y):
    final_y = []
    final_x = []
    for i in range(len(y)):
        final_y.append([])
        for j in range(len(x)):
            if (barier(x[j], y[i])):
                #if f(x[j], y[i]) > 50:
                    #print("i =", i, "j =", j)
                    #print("x =", x[j], "y =", y[i], "f =", f(x[j], y[i]))
                final_y[i].append(x[j])
            else: final_y[i].append(0)

    for i in range(len(x)):
        final_x.append([])
        for j in range(len(y)):
            if (barier(x[j], y[i])):
                final_x[i].append(y[j])
            else: final_x[i].append(0)

    #for i in range(len(final_x)):
    #    print(i,")", final_x[i])
    return final_y, final_x

def fill_z(x, y):
    z = []
    for i in range(len(x)):
        z.append([])
        for j in range(len(x[i])):
            if (x[i][j] != 0 and y[j][i] != 0):
                z[i].append(f(x[i][j], y[j][i]))
            else: z[i].append(0.0)
            #print("i =", i, "j =", j)
            #print("x =", x[i][j], "y =", y[j][i], "z =", z[i][j])
    
    #for i in range(len(z)):
    #    print(i,")", z[i])
    r = np.array(z)
    #for i in range(len(z)):
    #    r.__add__(np.array[z[i]])
    return r

def fill_F2(x, y):
    z = []
    for i in range(len(x)):
        z.append([])
        for j in range(len(x[i])):
            if (barier(x[i][j], y[i][j])):
                z[i].append(f(x[i][j], y[i][j]))
            else: z[i].append(0.0)

    r = np.array(z)
    #for i in range(len(z)):
    #    r.__add__(np.array[z[i]])
    #print(r)
    return r

def g1_1(x1):
    return (-3*x1 + 6) / 2

def g2_1(x1):
    return (-x1 - 3) / (-1)

def g3_1(x1):
    return (x1 - 7) / (-1)

def g4_1(x1):
    return (2*x1 - 4) / 3


def drawBoder(ax, x1, g):
    zs = np.arange(0, 80, 35)

    X, Z = np.meshgrid(x1, zs)
    Y = g(X)

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, alpha = 0.4)


def show(x1_list, x2_list):
    N = int(x1_list.__len__())
    if (N <= 0):
        return

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1,x2)')
    #x1_array = np.arange(min(x1_list) - 0.1, max(x1_list) + 0.1, 0.1)
    #x2_array = np.arange(min(x2_list) - 0.1, max(x2_list) + 0.1, 0.1)
    #x1_array, x2_array = np.meshgrid(x1_array, x2_array)
    #R = f(x1_array, x2_array) 
    #ax.plot_surface(x1_array, x2_array, R, color='b', alpha=0.5) 
    drawFunc(0, 0, 5, 5, ax)

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

# <---------- f
def f(x1, x2):
    return (x1-6)**2 +(x2-7)**2

def f_x1(x1, x2):
    return 2*x1 - 12

def f_x2(x1, x2):
    return 2*x2 - 14
# -------------->

# <---------- gi
def g1(x1, x2):
    return -3*x1 - 2*x2 + 6

def g2(x1, x2):
    return -x1 + x2 - 3

def g3(x1, x2):
    return x1 + x2 - 7

def g4(x1, x2):
    return 2*x1 - 3*x2 - 4
# -------------->


# <---------- gi_bool
def g1_bool(x1, x2):
    return -3*x1 - 2*x2 + 6 <= 0

def g2_bool(x1, x2):
    return -x1 + x2 - 3 <= 0

def g3_bool(x1, x2):
    return x1 + x2 - 7 <= 0

def g4_bool(x1, x2):
    return 2*x1 - 3*x2 - 4 <= 0

def barier(x1, x2):
    return (g1_bool(x1, x2) and g2_bool(x1, x2) and g3_bool(x1, x2) and g4_bool(x1, x2))

# -------------->

# <---------- X
def F(x1, x2, r):
    return f(x1,x2) + P(x1, x2, r)

def F_x1(x1, x2, r):
    return f_x1(x1, x2) + P_x1(x1, x2, r)

def F_x2(x1, x2, r):
    return f_x2(x1, x2) + P_x2(x1, x2, r)
# -------------->


# <-------------- P
def P(x1, x2, r):
    sum = 1/g1(x1, x2) + 1/g2(x1, x2) + 1/g3(x1, x2) + 1/g4(x1, x2)
    return -r*sum

def P_x1(x1, x2, r):
    sum = 3/(g1(x1, x2)**2) + 1/(g2(x1, x2)**2) - 1/(g3(x1, x2)**2) - 1/(g4(x1, x2)**2)
    return -r*sum

def P_x2(x1, x2, r):
    sum = 2/(g1(x1, x2)**2) - 1/(g2(x1, x2)**2) - 1/(g3(x1, x2)**2) + 3/(g4(x1, x2)**2)
    return -r*sum

# ------------>

def gradient(x1, x2, r):
    i = F_x1(x1, x2, r)
    j = F_x2(x1, x2, r)
    return [i, j]

def module_of_gradient(grad):
    i = 0; j = 1
    return sqrt(grad[i]**2 + grad[j]**2)


def method_of_gradient_descent_with_a_constant_step(x1, x2, e, M, r):
    global counter
    k = 0
    counter += 1
    x1_next = x1
    x2_next = x2
    while True:
        counter += 2
        grad = gradient(x1, x2, r)
        module_grad = module_of_gradient(grad)
        
        if ((module_grad < e) and (k >= M)):
            return (x1_next, x2_next)

        gamma = 0.1
        x1_next = x1 - gamma * grad[0]
        x2_next = x2 - gamma * grad[1]
        
        counter += 2
        while (F(x1_next, x2_next, r) - F(x1, x2, r) >= 0 or not barier(x1_next, x2_next)):
            gamma /= 4
            x1_next = x1 - gamma * grad[0]
            x2_next = x2 - gamma * grad[1]
            counter += 1
        
        #print(grad, 'x1 =', x1, 'x2 =', x2, 'x1_next =', x1_next, 'x2_next =', x2_next, 'gamma =', gamma)
        
        x1_list.append(x1); x2_list.append(x2)
        if ((sqrt(abs(x1_next - x1)**2 + abs(x2_next - x2)**2) <= e)
            & (abs(F(x1_next, x2_next, r) - F(x1, x2, r)) <= e)):
            return (x1_next, x2_next)

        x1 = x1_next
        x2 = x2_next
        k += 1


def barrier_function_method(x1, x2, r, C, e, M, k):
    min_x1, min_x2 = method_of_gradient_descent_with_a_constant_step(x1, x2, e, M, r)
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
    return barrier_function_method(min_x1, min_x2, r, C, e, M, k)


round_num = 4
x1 = 2.5
x2 = 1
e = 0.0001
M = 100
r = 1
c = 10
k = 0

result = barrier_function_method(x1, x2, r, c, e, M, k)
print(f"Barrier function method: {result[0]}; count of iteractions = {result[1]}")
print('Count of compute function =', counter + 1)


show(x1_list, x2_list)
#drawFunc(0, 0, 5, 5)
