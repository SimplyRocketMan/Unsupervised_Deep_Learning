import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from util import getKaggleMNIST

def main():
    xtr, ytr, xts, yts = getKaggleMNIST()

    PCA_ = PCA()
    reduced = PCA_.fit_transform(xtr)
    print(xtr[:])
    print(xtr.shape)
    plt.figure()
    plt.scatter(xtr[:,0], xtr[:,1], s=70, alpha=0.7)
    plt.title("Image")
    plt.figure()
    plt.scatter(reduced[:,0],reduced[:,1],s=70,c=ytr,alpha=0.5)
    plt.title("Image transformed")
    plt.show()

    plt.plot(PCA_.explained_variance_ratio_) # i.e. eigenvalues
    plt.title("Variance/Eigenvalues")
    plt.show()

    cumulative_variance = []
    last = 0
    for v in PCA_.explained_variance_ratio_:
        cumulative_variance.append(last+v)
        last = cumulative_variance[-1]
    plt.figure()
    plt.plot(cumulative_variance)
    plt.title("Cumulative variance")
    plt.show()

if __name__ == "__main__":
    main()
