from math import exp
from math import sin


def f(x):
    return (x-1)**2/x



def uniform_search_method(a, b, epsilon):
    iter = 0
    x = a
    while (x < b - epsilon):
        if (f(x) < f(x + epsilon)):
            return [(round(x, 7), round(f(x), 7)), iter]
        x += epsilon
        iter += 1



def dichotomy_mehod(a, b, epsilon, iter = 0):
    x = (a + b) / 2
    
    if (f(x - epsilon) < f(x + epsilon)):
        b = x
    else:
        a = x

    if(abs(b - a) >= 2 * epsilon):
        return dichotomy_mehod(a, b, epsilon, iter + 1)
        
    return [(round(x, 7), round(f(x), 7)), iter]




def fibonacci_sum(n):
    return fib_iter(1, 0, n)

def fib_iter(a, b, count):
    if (count == 0):
        return b
    return fib_iter(a + b, a, count - 1)

def find_fibonacci_number(N):
    iter = 1
    while(fibonacci_sum(iter) < N):
        iter += 1
    return iter

def fibonacci_method(a, b, epsilon):
    iter = 0
    S = find_fibonacci_number((b - a) / epsilon)
    K = 1
    Fs = fibonacci_sum(S)
    l = 1/Fs * (b-a)
    x1 = a + l * fibonacci_sum(S - 1 - K)
    x2 = b - l * fibonacci_sum(S - 1 - K)
    A = f(x1)
    B = f(x2)

    while (True):
        if (A < B):
            b = x2
        else:
            a = x1

        K += 1
        if (K == S - 1):
            break
        iter += 1

        if (A < B):
            x2 = x1
            B = A
            x1 = a + l * fibonacci_sum(S - 1 - K)
            A = f(x1)
        else:
            x1 = x2
            A = B
            x2 = b - l * fibonacci_sum(S - 1 - K)
            B = f(x2)
    x2 = x1 + epsilon
    B = f(x2)

    if (A < B):
        b = x1
    else:
        a = x1
    x = (a + b)/2
    return [(round(x, 7), round(f(x), 7)), iter]

print("x = 1.0, y = ", f(1))

print("Uniform search method: ", uniform_search_method(0.5, 2, 0.0001))

print("Dichotomy mehod: ", dichotomy_mehod(0.5, 2, 0.0001))

print("Fibonachy mehod: ", fibonacci_method(0.5, 2, 0.0001))
