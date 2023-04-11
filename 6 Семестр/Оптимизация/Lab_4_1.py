from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x1_list = []
x2_list = []
y_list = []
counter = 0

def show(x1_list, x2_list):
    N = int(x1_list.__len__())
    if (N <= 0):
        return

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    x1_array = np.arange(min(x1_list) - 1, max(x1_list) + 1, 0.01)
    x2_array = np.arange(min(x2_list) - 1, max(x2_list) + 1, 0.01)
    #x1_array = np.arange(-6, 3, 0.1)
    #x2_array = np.arange(-6, 6, 0.1)

    x1_array, x2_array = np.meshgrid(x1_array, x2_array)
    R = f(x1_array, x2_array)

    fig = plt.figure()
    ax = Axes3D(fig)


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
    #print(x1_list[0], x2_list[0], f(x1_list[0], x2_list[0]))

    for n in range(1, N):

        ax.scatter(x1_list[n], x2_list[n], f(x1_list[n], x2_list[n]), c='red')
        x1_list2.append(x1_list[n])
        x2_list2.append(x2_list[n])
        f_list.append(f(x1_list[n], x2_list[n]))
        #print(x1_list[n], x2_list[n], f(x1_list[n], x2_list[n]))
        
        fig.canvas.draw()
        #fig.canvas.flush_events()

    ax.scatter(x1_list[N - 1], x2_list[N - 1], f(x1_list[N - 1], x2_list[N - 1]), c='green')
    #print(x1_list[N - 1], x2_list[N - 1], f(x1_list[N - 1], x2_list[N - 1]))

    ax.plot(x1_list2, x2_list2, f_list, color="black")

    plt.show()

def f(x1, x2):
    return 3*x1**4 - x1*x2 + x2**4 - 7*x1 - 8*x2 + 2
def f_x1(x1, x2):
    return 12*x1**3 - x2 - 7
def f_x2(x1, x2):
    return 4*x2**3 - x1  - 8

def gradient(x1, x2):
    i = f_x1(x1, x2)
    j = f_x2(x1, x2)
    return [i, j]


def module_of_gradient(grad):
    i = 0; j = 1
    return sqrt(grad[i]**2 + grad[j]**2)

def dichotomy_mehod(a, b, epsilon, x1, x2, d1, d2):
    x = (a + b) / 2
    global counter
    counter += 2
    
    if (f(x1 + (x - epsilon)* d1, x2 + (x - epsilon)* d2) < f(x1 + (x + epsilon)* d1, x2 + (x + epsilon)* d2)):
        b = x
    else:
        a = x
            
    if(abs(b - a) >= 2 * epsilon):
        return dichotomy_mehod(a, b, epsilon, x1, x2, d1, d2)
    return x


def the_fletcher_reevse_method(x1, x2, e1, e2, M):
    global counter
    k = 0
    d_prev = [0, 0]
    grad_prev = 0
    while True:
        counter += 2
        grad = gradient(x1, x2)
        module_grad = module_of_gradient(grad)
        if ((module_grad < e1) | (k >= M)):
            return [(round(x1, round_num), round(x2, round_num), round(f(x1, x2), round_num)), k]

        B = 0

        if k % 2 == 1: B = module_of_gradient(grad)**2 / module_of_gradient(grad_prev)**2

        d = [-grad[0] + B * d_prev[0], -grad[1] + B * d_prev[1]] 
        t = dichotomy_mehod(0, 0.1, e2, x1, x2, d[0], d[1])

        x1_next = x1 - t * grad[0]
        x2_next = x2 - t * grad[1]

        x1_list.append(x1); x2_list.append(x2)
    
        counter += 1
        if ((sqrt(abs(x1_next - x1)**2 + abs(x2_next - x2)**2) <= e2)
            & (abs(f(x1_next, x2_next) - f(x1, x2)) <= e2)):
            return [(round(x1_next, round_num), 
                     round(x2_next, round_num), 
                     round(f(x1_next, x2_next), round_num)), 
                    k]

        x1 = x1_next; x2 = x2_next
        d_prev = d; grad_prev = grad
        k += 1


round_num = 3
x1 = 10
x2 = 10
e1 = 0.001
e2 = 0.001
M = 100

result = the_fletcher_reevse_method(x1, x2, e1, e2, M)
print(f"Newton's method with step adjustment: {result[0]}; count of iteractions = {result[1]}")
print('Count of compute function =', counter)


show(x1_list, x2_list)
