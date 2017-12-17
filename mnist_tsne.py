import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from util import getKaggleMNIST


def main():
    X,Y,Xt,Yt = getKaggleMNIST()
    sample_size = 500
    print(X[0:sample_size].shape, Y[0:sample_size].shape)
    #plt.scatter(X[0:10,0],X[0:10,1], s=70,c=Y[0:10],alpha=.5)
    #plt.show()
    tsne = TSNE()

    Z = tsne.fit_transform(X[0:sample_size], Y[:sample_size])
    plt.subplot(211)
    plt.scatter(Z[:,0],Z[:,1],s=70,c=Y[:sample_size], alpha=.5)
    plt.colorbar()
    Z1 = tsne.fit_transform(Z)
    plt.subplot(212)
    plt.scatter(Z1[:,0],Z1[:,1],s=70, c=Y[:sample_size], alpha=.5)
    plt.colorbar()
    plt.show()

    
if __name__ == "__main__":
    main()
