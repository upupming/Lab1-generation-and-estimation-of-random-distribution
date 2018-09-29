import numpy
from matplotlib import pyplot

def guassian_cal(x, mu, sigma):
    """Calculate Guassian probability, see equation (11) (D = 2: [密度, 含糖率])  """
    return numpy.exp(-0.5 * 
        numpy.matmul(
            numpy.matmul(
                numpy.transpose(x - mu), numpy.linalg.inv(sigma)
            ),
            x - mu
        )
    ) / ((2 * numpy.pi) * numpy.sqrt(numpy.abs(numpy.linalg.det(sigma))))

def em(N, S, I):
    print(f'EM algorithm with N = {N}, data = {data} and I = {I} is running...')
    with open(f'../generate_results/em-{I}.txt', 'w+') as generate_results:
            generate_results.write(f'N = {N}\n\n')

    # Init params, see equation (16)
    k = 3
    alpha = numpy.full(k, 1.0/3)
    mu = numpy.empty((3, 2))
    mu[0] = S[:,5]
    mu[1] = S[:,21]
    mu[2] = S[:,26]
    sigma = numpy.empty((3, 2, 2))
    sigma[0] = sigma[1] = sigma[2] = numpy.array(
        [
            [0.1, 0.0],
            [0.0, 0.1]
        ]
    )
    
    # Posterior probability matrix
    Y = numpy.empty((N, k))
    # Iterating, see equation (13) and(14)
    for iter_time in range(I):
        for i in range(k):
            for j in range(N):
                # See equation (13)
                Y[j][i] = alpha[i] * guassian_cal(S[:,j], mu[i], sigma[i]) / (numpy.dot(alpha, [
                    guassian_cal(S[:,j], mu[0], sigma[0]),
                    guassian_cal(S[:,j], mu[1], sigma[1]),
                    guassian_cal(S[:,j], mu[2], sigma[2])
                    ]))
        
        # See equationn (14) 
        alpha = [
            numpy.mean(Y[:,0]),
            numpy.mean(Y[:,1]),
            numpy.mean(Y[:,2])
        ]
        mu = [
            numpy.average(S, axis=1, weights=Y[:,0]),
            numpy.average(S, axis=1, weights=Y[:,1]),
            numpy.average(S, axis=1, weights=Y[:,2])
        ]
        temp = numpy.zeros((3, 2, 2))
        for j in range(N):
            temp[0] += Y[j][0] * numpy.matmul((S[:,j] - mu[0]).reshape(-1, 1), (S[:,j] - mu[0]).reshape(1, -1))
            temp[1] += Y[j][1] * numpy.matmul((S[:,j] - mu[1]).reshape(-1, 1), (S[:,j] - mu[1]).reshape(1, -1))
            temp[2] += Y[j][2] * numpy.matmul((S[:,j] - mu[2]).reshape(-1, 1), (S[:,j] - mu[2]).reshape(1, -1))
        sigma = [
            temp[0] / numpy.sum(Y[:,0]),
            temp[1] / numpy.sum(Y[:,1]),
            temp[2] / numpy.sum(Y[:,2]),
        ]

        # Save alpha, mu, and sigma to file
        with open(f'../generate_results/em-{I}.txt', 'a+') as generate_results:
            generate_results.write('##### ')
            generate_results.write(f'iter = {iter_time + 1}')
            generate_results.write(' #####\n\n')

            generate_results.write(f'Y = {Y}\n\n')

            generate_results.write(f'alpha[0] = {alpha[0]}\n')
            generate_results.write(f'alpha[1] = {alpha[1]}\n')
            generate_results.write(f'alpha[2] = {alpha[2]}\n\n')

            generate_results.write(f'mu[0] = {mu[0]}\n')
            generate_results.write(f'mu[1] = {mu[1]}\n')
            generate_results.write(f'mu[2] = {mu[2]}\n\n')

            generate_results.write(f'sigma[0] = {sigma[0]}\n')
            generate_results.write(f'sigma[1] = {sigma[1]}\n')
            generate_results.write(f'sigma[2] = {sigma[2]}\n\n')

    N_array = numpy.arange(1, N+1)

    pyplot.rcParams['font.family'] = 'sans-serif'
    pyplot.rcParams['font.sans-serif'] = ['SimHei', 'Helvetica', 'Calibri']
    pyplot.xlabel('密度')
    pyplot.ylabel('含糖量')
    pyplot.title(f'EM 算法聚类结果\n 迭代次数 $I = {I}$')
    # Splict classes
    C = numpy.empty(N)
    for n in range(N):
        if Y[n][0] >= Y[n][1] and Y[n][0] >= Y[n][2]:
            C[n] = 0
            pyplot.plot(S[0,n], S[1, n], 'ro')
        elif Y[n][1] >= Y[n][0] and Y[n][1] >= Y[n][2]:
            C[n] = 1
            pyplot.plot(S[0,n], S[1, n], 'go')
        else:
            C[n] = 2
            pyplot.plot(S[0,n], S[1, n], 'bo')

    # Save to images
    pyplot.savefig(f'../generate_results/em-{I}.png', bbox_inches='tight')
    pyplot.close()


N = 30
data = numpy.array(
    [
        # Row 1: 密度
        [0.697, 0.774, 0.634, 0.608, 0.556, 0.403, 0.481, 0.437, 0.666, 0.243, 0.245, 0.343, 0.639, 0.657, 0.360, 0.593, 0.719, 0.359, 0.339, 0.282, 0.748, 0.714, 0.483, 0.478, 0.525, 0.751, 0.532, 0.473, 0.725, 0.446],
        # Row 2: 含糖率
        [0.460, 0.376, 0.264, 0.318, 0.215, 0.237, 0.149, 0.211, 0.091, 0.267, 0.057, 0.099, 0.161, 0.198, 0.370, 0.042, 0.103, 0.188, 0.241, 0.257, 0.232, 0.346, 0.312, 0.437, 0.369, 0.489, 0.472, 0.376, 0.445, 0.459]
    ]
)

em(N, data, 50)