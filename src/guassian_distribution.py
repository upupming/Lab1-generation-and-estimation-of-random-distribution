import numpy
from matplotlib import pyplot

def generate_guassian(mu, squareOfSigma, size):
    size = int(size)
    print(f'Guassian distribution with mu = {mu}, sigma^2 = {squareOfSigma} and size = {size} is running...')

    sigma = numpy.sqrt(squareOfSigma)
    data = numpy.random.normal(mu, sigma, size)
    
    N = numpy.arange(1, size+1)
    E = numpy.empty(size)
    D = numpy.empty(size)
    E[0] = data[0]
    D[0] = 0
    # Calculate D & E in interval [1, size)
    for n in range(1, size):
        E[n] = (E[n-1]*n + data[n])/(n+1)
        D[n] = numpy.std(data[0:n])

    pyplot.rcParams['font.family'] = 'sans-serif'
    pyplot.rcParams['font.sans-serif'] = ['SimHei', 'Helvetica', 'Calibri']
    pyplot.xlabel('n')
    pyplot.title(f'正态分布样本的均值、方差随样本数增加而变化的图像\n $mu = {mu}, \sigma ^2 = {squareOfSigma}, N = {size}$')
    lineE, = pyplot.plot(N, E, 'r', label='均值 E')
    lineD, = pyplot.plot(N, D, 'g', label='方差 D')
    pyplot.legend(handles=[lineE, lineD])

    # Save to images
    pyplot.savefig(f'../generate_results/guassian-{size}.png', bbox_inches='tight')
    pyplot.close()


generate_guassian(10.0, 5.0, 1e3)
generate_guassian(10.0, 5.0, 1e4)