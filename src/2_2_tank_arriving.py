import numpy

def num_of_arrivial(lam, N, size):
    size = int(size)
    print(f'Number of arraival with lambda = {lam}, N = {N} and size = {size} is running...')
    
    A = numpy.empty(N)
    A_total = 0
    for n in range(N):
        T = numpy.random.poisson(lam, size)
        E = numpy.mean(T)
        A[n] = E
        A_total += E
        
    # Save to file
    with open(f'../generate_results/tank-num-of-arrival-{size}.txt', 'w+') as generate_results:
        generate_results.write(f'size = {size}\n')
        generate_results.write(f'lambda = {lam}\n')
        generate_results.write(f'N = {N}\n\n')
        generate_results.write(f'A = {A}\n')
        generate_results.write(f'A_total = {A_total}\n')

def time_of_arrival(lam, N, M, size):
    M = int(M)
    size = int(size)
    print(f'Time of arraival with lambda = {lam}, N = {N}, M = {M} and size = {size} is running...')

    B = numpy.empty(M)
    B_new = numpy.empty(M)
    for m in range(M):
        S = numpy.random.exponential(1/lam)
        E = numpy.mean(S)
        B[m] = E
        B_new[m] = numpy.sum(B[0:m])
        if(B_new[m] > N): 
            B_new = B_new[0:m]
            break

    # Save to file
    with open(f'../generate_results/tank-time-of-arrival-{M}.txt', 'w+') as generate_results:
        generate_results.write(f'size = {size}\n')
        generate_results.write(f'lambda = {lam}\n')
        generate_results.write(f'M = {M}\n')
        generate_results.write(f'N = {N}\n\n')
        generate_results.write(f"B' = {B_new}\n")
        generate_results.write(f"size of B' = {len(B_new)}\n")


num_of_arrivial(4.0, 3, 1e3)
num_of_arrivial(4.0, 3, 1e4)
num_of_arrivial(4.0, 3, 1e5)
num_of_arrivial(4.0, 3, 1e6)

time_of_arrival(4.0, 3, 1e3, 1e6)
time_of_arrival(4.0, 3, 1e4, 1e6)
time_of_arrival(4.0, 3, 1e5, 1e6)
time_of_arrival(4.0, 3, 1e6, 1e6)
