from math import exp

#def f(x):
#    return x**4 + exp(-x)

#def f_1(x):
#    return 4*x**3 - exp(-x)

# a = 0 b = 1

def f(x):
    return 5/x + x*x

def f_1(x):
    return -5/(x*x) + 2*x


a = 0.5
b = 2
eps = 0.001
round_num = 3


def the_midpoint_method(a, b, e, iter = 0):
    iter += 1

    x = (a+b)/2
    f_1_value = f_1(x)

    if(abs(f_1_value) <= e):
        return [(round(x, round_num), round(f(x), round_num)), iter]

    if(f_1_value > 0):
        return the_midpoint_method(a, x, e, iter)
    
    return the_midpoint_method(x, b, e, iter)


def the_chord_method(a, b, e):
    iter = 0
    f_1_a_value = f_1(a)
    f_1_b_value = f_1(b)
    f_1_x_value = 0

    if(f_1_a_value * f_1_b_value < 0):
        
        def inner_counting():
            nonlocal f_1_a_value, f_1_b_value, f_1_x_value, a, b, e, iter
            iter += 1

            x = a - f_1_a_value / (f_1_a_value - f_1_b_value) * (a - b)
            f_1_x_value = f_1(x)
            
            #print(round(x, round_num), round(a, round_num), round(b, round_num), round(f_1_a_value, round_num), round(f_1_b_value, round_num), round(f_1_x_value, round_num + 1))

            if(abs(f_1_x_value) <= e):
                return [(round(x, round_num), round(f(x), round_num)), iter]
            
            if(f_1_x_value > 0):
                b = x
                f_1_b_value = f_1_x_value
                return inner_counting()
            
            a = x
            f_1_a_value = f_1_x_value
            return inner_counting()
        
        return inner_counting()

    if(f_1_a_value > 0 and f_1_b_value > 0):
        return [(round(a, round_num), round(f(a), round_num)), iter]

    if(f_1_a_value < 0 and f_1_b_value < 0):
            return [(round(b, round_num), round(f(b), round_num)), iter]

    if(f_1_a_value == 0):
        return [(round(a, round_num), round(f(a), round_num)), iter]

    if(f_1_b_value == 0):
        return [(round(b, round_num), round(f(b), round_num)), iter]
    
    return 0



result = the_midpoint_method(a, b, eps)
print(f"The midpoint method: {result[0]}; count of iteractions = {result[1]}")

result = the_chord_method(a, b, eps)
print(f"The chord method:: {result[0]}; count of iteractions = {result[1]}")