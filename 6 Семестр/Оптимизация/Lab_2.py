
def f(x):
    return 5/x + x*x

def f_1(x):
    return -5/(x*x) + 2*x

def f_2(x):
    return 10/(x*x*x) + 2


a = 0.5
b = 2
eps = 0.001
round_num = 3


def the_midpoint_method(a, b, e, iter = 0):
    x = (a+b)/2
    f_1_value = f_1(x)

    if(abs(f_1_value) <= e):
        return [(round(x, round_num), round(f(x), round_num)), iter]
    
    iter += 1

    if(f_1_value > 0):
        b = x
        return the_midpoint_method(a, b, e, iter)
    
    a = x
    return the_midpoint_method(a, b, e, iter)


def the_chord_method(a, b, e, iter = 0):
    f_1_a_value = f_1(a)
    f_1_b_value = f_1(b)
    f_1_x_value = 0

    def inner_counting():
        nonlocal f_1_a_value, f_1_b_value, f_1_x_value, a, b, e, iter

        if(f_1_a_value * f_1_b_value < 0):
            x = a - f_1_a_value / (f_1_a_value - f_1_b_value) * (a - b)
            f_1_x_value = f_1(x)
            
            if(abs(f_1_x_value) <= e):
                return [(round(x, round_num), round(f(x), round_num)), iter]
            
            iter += 1

            if(f_1_x_value > 0):
                b = x
                f_1_b_value = f_1_x_value
                return inner_counting()
            
            a = x
            f_1_a_value = f_1_x_value
            return inner_counting()

        if(f_1_a_value == 0):
            return [(round(a, round_num), round(f(a), round_num)), iter]

        if(f_1_b_value == 0):
            return [(round(b, round_num), round(f(b), round_num)), iter]

        if(f_1_a_value > 0):
            return [(round(a, round_num), round(f(a), round_num)), iter]

        if(f_1_a_value < 0):
            return [(round(b, round_num), round(f(b), round_num)), iter]

    return inner_counting()

result = the_midpoint_method(a, b, eps)
print(f"The midpoint method: {result[0]}; count of iteractions = {result[1]}")

result = the_chord_method(a, b, eps)
print(f"The chord method:: {result[0]}; count of iteractions = {result[1]}")