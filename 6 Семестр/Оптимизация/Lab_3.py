from math import sqrt
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x1_list = []
x2_list = []
y_list = []
counter_1 = 0
counter_2 = 0


def f(x1, x2):
    return 100 * (x2 - x1**2)**2 + (1 - x1)**2
def f_x1(x1, x2):
    return 400 * x1**3 + (2 - 400*x2)*x1 - 2
def f_x2(x1, x2):
    return 200 * x2 - 200 * x1**2

def gradient(x1, x2):
    i = f_x1(x1, x2)
    j = f_x2(x1, x2)
    return [i, j]

def module_of_gradient(grad):
    i = 0; j = 1
    return sqrt(grad[i]**2 + grad[j]**2)


def method_of_gradient_descent_with_a_constant_step(x1, x2, e, M):
    global counter_1
    k = 0
    counter_1 += 1
    while True:
        counter_1 += 2
        grad = gradient(x1, x2)
        module_grad = module_of_gradient(grad)
        
        if ((module_grad < e) | (k >= M)):
            return [(round(x1, round_num), round(x2, round_num), round(f(x1, x2), round_num)), k]

        #gamma = 0.018
        gamma = 0.5
        x1_next = x1 - gamma * grad[0]
        x2_next = x2 - gamma * grad[1]
        
        counter_1 += 2
        while (f(x1_next, x2_next) - f(x1, x2) >= 0):
            gamma /= 2
            x1_next = x1 - gamma * grad[0]
            x2_next = x2 - gamma * grad[1]
            counter_1 += 1
        
        #print(grad, 'x1 =', x1, 'x2 =', x2, 'x1_next =', x1_next, 'x2_next =', x2_next, 'gamma =', gamma)
        
        x1_list.append(x1); x2_list.append(x2)
        if ((sqrt(abs(x1_next - x1)**2 + abs(x2_next - x2)**2) <= e*e)
            & (abs(f(x1_next, x2_next) - f(x1, x2)) <= e*e)):
            return [(round(x1_next, round_num), 
                     round(x2_next, round_num), 
                     round(f(x1_next, x2_next), round_num)), 
                    k]

        x1 = x1_next
        x2 = x2_next
        k += 1

# ------------------------------------------------------------------------------------

def f_lambda(x1, x2, l, dx1f, dx2f):
    return f(x1 - l*dx1f, x2 - l*dx2f)

def dichotomy_mehod(a, b, epsilon, x1, x2, dx1f, dx2f):
    l = (a + b) / 2
    global counter_2
    counter_2 += 2

    if (f_lambda(x1, x2, l - epsilon, dx1f, dx2f) < f_lambda(x1, x2, l + epsilon, dx1f, dx2f)):
        b = l
    else:
        a = l

    if(abs(b - a) >= 2 * epsilon):
        return dichotomy_mehod(a, b, epsilon, x1, x2, dx1f, dx2f)
    return l

def method_of_the_steepest_gradient_descent(x1, x2, e1, e2, M):
    global counter_2
    k = 0
    while True:
        counter_2 += 2
        grad = gradient(x1, x2)
        module_grad = module_of_gradient(grad)
        if ((module_grad < e1) | (k >= M)):
            return [(round(x1, round_num), round(x2, round_num), round(f(x1, x2), round_num)), k]

        gamma = dichotomy_mehod(0, 0.1, e2, x1, x2, grad[0], grad[1])

        x1_next = x1 - gamma * grad[0]
        x2_next = x2 - gamma * grad[1]

        #print(grad, 'x1 =', x1, 'x2 =', x2, 'gamma =', gamma)

        #x1_list.append(x1); x2_list.append(x2)
        if ((sqrt(abs(x1_next - x1)**2 + abs(x2_next - x2)**2) <= e2)
            & (abs(f(x1_next, x2_next) - f(x1, x2)) <= e2)):
            return [(round(x1_next, round_num), 
                     round(x2_next, round_num), 
                     round(f(x1_next, x2_next), round_num)), 
                    k]

        x1 = x1_next
        x2 = x2_next
        k += 1

def show(x1_list, x2_list):
    N = x1_list.__len__()
    if (N <= 0):
        return

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    
    #x1_array = np.arange(min(x1_list) - 1, max(x1_list) + 1, 0.01)
    #x2_array = np.arange(min(x2_list) - 1, max(x2_list) + 1, 0.01)

    x1_array = np.arange(-1.2, 1.2, 0.01)
    x2_array = np.arange(0.2, 1.2, 0.01)

    x1_array, x2_array = np.meshgrid(x1_array, x2_array)
    R = f(x1_array, x2_array)
    
    #plt.ion()   # включение интерактивного режима отображения графиков
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.plot_surface(x1_array, x2_array, R, color='b', alpha=0.5)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1,x2)')
    r = 1
    
    if (N > 100):
        r = 10
    if (N > 1000):
        r = 50

    x1_list2 = []
    x2_list2 = []
    f_list = []

    ax.scatter(x1_list[0], x2_list[0], f(x1_list[0], x2_list[0]), c='black')
    x1_list2.append(x1_list[0])
    x2_list2.append(x2_list[0])
    f_list.append(f(x1_list[0], x2_list[0]))

    #ax.scatter(x1_list[1], x2_list[1], f(x1_list[1], x2_list[1]), c='red')
    #x1_list2.append(x1_list[1])
    #x2_list2.append(x2_list[1])
    #f_list.append(f(x1_list[1], x2_list[1]))

    for n in range(r, N - 1*r, r):

        ax.scatter(x1_list[n], x2_list[n], f(x1_list[n], x2_list[n]), c='red')
        x1_list2.append(x1_list[n])
        x2_list2.append(x2_list[n])
        f_list.append(f(x1_list[n], x2_list[n]))

        #fig.canvas.draw()
        #fig.canvas.flush_events()
        #time.sleep(0.001)
    #fig.canvas.draw()
    ax.scatter(x1_list[N - 1], x2_list[N - 1], f(x1_list[N - 1], x2_list[N - 1]), c='green')

    ax.plot(x1_list2, x2_list2, f_list, color="black")

    #plt.ioff()   # выключение интерактивного режима отображения графиков

    plt.show()


x1 = 0.5
x2 = 0.5
e1 = 0.001
e2 = 0.001 * 0.001
round_num = 4
M = 10000

result = method_of_gradient_descent_with_a_constant_step(x1, x2, e1, M)
print(f"The method of gradient descent with a constant step: {result[0]}; count of iteractions = {result[1]}")
print('Count of compute function =', counter_1)

result = method_of_the_steepest_gradient_descent(x1, x2, e1, e2, M)
print(f"The method of the steepest gradient descent: {result[0]}; count of iteractions = {result[1]}")
print('Count of compute function =', counter_2)

show(x1_list, x2_list)