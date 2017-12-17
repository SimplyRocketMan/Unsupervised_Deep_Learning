import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

def get_xor_data():
    N = 100
    
    centers = [[0,0],
         [1,1],
         [1,0],
         [0,1]]
    print(np.array(centers).shape)
    X = []
    for i in range(4):
        X.append(np.random.random((N,2)) - centers[i]) 
    X = np.vstack([X[0],X[1],X[2],X[3]])
    Y = np.array([0]*(2*N)+[1]*(2*N))

    return np.array(X),np.array(Y)

def main():
    x,y = get_xor_data()
    #print(x[:,1],y)
    #print(x.shape)
    plt.scatter(x[:,0], x[:,1], s=70, c=y, alpha=0.5)
    #plt.show()

    tsne = TSNE()

    Z = tsne.fit_transform(x)
    plt.figure()
    plt.scatter(Z[:,0],Z[:,1],s=70,c=y,alpha=.5)
    plt.show()
    
if __name__=="__main__":
    main()
    
