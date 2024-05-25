from aleatory.processes import BrownianMotion
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
# plt.style.use("https://raw.githubusercontent.com/quantgirluk/matplotlib-stylesheets/main/quant-pastel-light.mplstyle")


## This is done by aleatory library
def easy_version():
    brownian = BrownianMotion()
    brownian.draw(n=100, N=200)

def pdf():
    process = BrownianMotion()
    W_1 = process.get_marginal(t=1) ## getting the marginal pdf
    x = np.linspace(-10,10,100)
    plt.plot(x,W_1.pdf(x),lw=1.5, alpha=0.75, label=f'$t$={1:.2f}')
    plt.title(f'$W_1$ pdf')
    plt.show()
    
def multi_pdf():
    process = BrownianMotion()
    fig, ax1 = plt.subplots(1,1)
    sigma = [1,2,5,10,100]
    for t in sigma:
        W_t = process.get_marginal(t)
        x = np.linspace(-30,30,500)
        ax1.plot(x,W_t.pdf(x),'-',lw=1.5,alpha=0.75, label= f'$t$={t:.2f}')
    ax1.legend()
    plt.title("$W_t$ pdfs")
    plt.show()
    
def sampling():
    process = BrownianMotion()
    W_t = process.get_marginal(t=1)
    sample = W_t.rvs(size=10)
    print(sample)

def Simulation():
    T = 1
    n = 100
    times = np.linspace(0,T,n)
    sigma = np.sqrt(T/(n-1))
    normal_increment = norm.rvs(loc = 0,scale=sigma, size = n-1)
    normal_increment = np.insert(normal_increment,0,0)
    W = normal_increment.cumsum()
    
    plt.plot(times, W, "-",lw=1.5)
    plt.show()
    
