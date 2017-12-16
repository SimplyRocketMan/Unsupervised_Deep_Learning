import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE


def Create3Dfigure():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    return fig, ax



def main():
    fig, ax = Create3Dfigure()
    # data
    centers = np.array([
    [ 1,  1,  1],
    [ 1,  1, -1],
    [ 1, -1,  1],
    [ 1, -1, -1],
    [-1,  1,  1],
    [-1,  1, -1],
    [-1, -1,  1],
    [-1, -1, -1],
    ])*1.5
    points_per_cloud = 100
    d = []
    for c in centers:
        cloud = np.random.rand(points_per_cloud, 3)+c
        d.append(cloud)
    d = np.concatenate(d)
    X = d[:,0]
    Y = d[:,1]
    Z = d[:,2]
    labels = np.array([[i]*points_per_cloud for i in range(len(centers))]).flatten()
    print(d.shape)    
    ax.scatter(X,Y,Z,c=labels,marker="o")

    plt.show()

    #tSNE magic
    figg,axx= Create3Dfigure()
    tsne = TSNE()

    trf = tsne.fit_transform(d)
    print(trf.shape)
    plt.figure()
    plt.scatter(trf[:,0],trf[:,1], c=labels)
    plt.show()

    
if __name__ == "__main__":
    main()
