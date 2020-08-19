import math

def opt_naive(model, min_p = 0, stride = 0.01, N = 10000, print_it = False):
    
    f = lambda x: x * model(x)
    max_act = -math.inf
    p = min_p

    for i in range(N):
        max_tmp = f(p)

        if max_tmp < max_act:
            if print_it:
                print(i)
                
            return p

        max_act = max_tmp
        p += stride

    print("Price out of range, use a bigger N or stride")

    return "NA"


    

