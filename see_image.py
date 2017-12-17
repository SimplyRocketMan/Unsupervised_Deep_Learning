import numpy as np
import matplotlib.pyplot as plt
from util import getKaggleMNIST

def main():
    xtr, ytr, xts, yts = getKaggleMNIST()
    index = 0
    print(xtr[index].shape)
    print(xtr[index])
    im = xtr[index].reshape((28,28))
    #plt.figure()
    #map(lambda x:plt.scatter(x,np.linspace(0,1)),xtr[index])
    
    plt.figure()
    plt.imshow(im)
    plt.title("Label {}".format(ytr[index]))
    plt.show()
    
if __name__ == "__main__":
    main()
