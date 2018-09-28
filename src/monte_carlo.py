import numpy

def generate_data(start, end, size):
    """Return random floats in the half-open interval [start, end)."""
    return (end - start) * numpy.random.random_sample(size) + start

def is_in_circle(x, y, a):
    return x*x + y*y <= a*a /4

def cal_pi(size=1000):
    size = int(size)
    print(f'Monte Carlo with size = {size} is running...')
    
    a = 1

    X = generate_data(-a/2, a/2, size)
    Y = generate_data(-a/2, a/2, size)

    n = size
    m = 0
    for i in range(int(size)):
        if is_in_circle(X[i], Y[i], a):
            m = m + 1
    
    pi = 4*m/n

    # Save X and Y to file
    with open(f'../generate_results/monte-carlo-{size}.txt', 'w+') as generate_results:
        generate_results.write(f'n = {n}\n')
        generate_results.write(f'm = {m}\n\n')
        generate_results.write(f'pi = {pi}\n')

cal_pi(1e3)
cal_pi(1e4)
cal_pi(1e5)
cal_pi(1e6)