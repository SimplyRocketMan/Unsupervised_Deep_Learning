import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

def divide_int(x,y):
    return np.divide(x,y).astype('int')

def generate_donut_data():
    N = 600
    r_in = 10
    r_out= 20

    r1 = np.random.randn(divide_int(N,2))+r_in
    r2 = np.random.randn(divide_int(N,2))+r_out
    theta = 2*np.pi*np.random.random(divide_int(N,2))

    X_in = np.concatenate([[r1*np.cos(theta)],[r1*np.sin(theta)]]).T
    X_out = np.concatenate([[r2*np.cos(theta)],[r2*np.sin(theta)]]).T

    X = np.concatenate([X_in,X_out])
    Y = np.array([0]*divide_int(N,2)+ [1]*divide_int(N,2))
    return X,Y
    
def main():
    x,y = generate_donut_data()
    print(x.shape)
    print(y.shape)
    
    plt.scatter(x[:,0],x[:,1],s=70, c=y, alpha=0.7)

    tsne = TSNE()
    trans = tsne.fit_transform(x)

    print(trans.shape)
    plt.figure()
    plt.scatter(trans[:,0],trans[:,1], s=70, c=y, alpha=0.6)
    plt.show()
if __name__ == "__main__":
    main()
