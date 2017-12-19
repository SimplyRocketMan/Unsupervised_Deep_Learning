import pdb
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import util 

from sklearn.utils import shuffle

class Autoencoder:
    def __init__(self, Din, id_):
        self.Din = Din
        self.id = id_
        
    def fit(self, X, learning_rate=1e-1, momentum=0.7, epochs=1, batch_size=100):
        N,D = X.shape
        n_batches = int(N/batch_size)
        W = util.init_weights((D,self.Din))
        #print(W, "\n\n\n", W.shape)

        
        self.W = theano.shared(W, 'W_%s'%self.id)
        self.bias_h = theano.shared(np.zeros(self.Din),'bh_%s'%self.id)
        self.bias_o = theano.shared(np.zeros(D), 'bo_%s'%self.id)
        self.params=[self.W, self.bias_h, self.bias_o]
        self.forward=[self.W,self.bias_h]

        # for momentum
        self.W_change = theano.shared(np.zeros(W.shape),'cW_%s'%self.id)
        self.bias_h_change = theano.shared(np.zeros(self.Din),'cbh_%s'%self.id)
        self.bias_o_change = theano.shared(np.zeros(D),'cbo_%s'%self.id)
        self.params_change=[self.W_change,self.bias_h_change,self.bias_o_change]
        self.forward_change=[self.W_change,self.bias_h_change]

        #pdb.set_trace()
        
        X_in = T.matrix('X_%s'%self.id)
        Y = self.output(X_in)

        H = T.nnet.sigmoid(X_in.dot(self.W)+self.bias_h)
        self.hidden_op = theano.function(inputs=[X_in], outputs=H)
        self.predict   = theano.function(inputs=[X_in], outputs=Y)
        #cost = ((X_in-Y)*(X_in-Y)).sum()/N
        cost = -(X_in*T.log(Y) + (1-X_in)*T.log(1-Y)).sum()/N
        cost_op = theano.function(inputs=[X_in], outputs=cost)

        updates = [
            (p, p+momentum*dp - learning_rate*T.grad(cost,p)) for p,dp in zip(self.params, self.params_change)
            ]+[
                (dp, momentum*dp - learning_rate*T.grad(cost,p)) for p,dp in zip(self.params, self.params_change)
            ]
        train_op = theano.function(inputs=[X_in], updates=updates)

        costs = []
        #pdb.set_trace()
        print("Training autoencoder %s"%self.id)
        for i in range(epochs):
            print("Epoch ", i)
            X = shuffle(X)
            for j in range(n_batches):
                #pdb.set_trace()
                batch = X[j*batch_size:(j*batch_size +batch_size)]
                train_op(batch)
                cost_ = cost_op(X)
                print(j,"/",n_batches,"\nCost:",cost_)
                costs.append(cost_)

        plt.plot(costs)
        plt.show()
        
    def forward_hidden(self, X):
        return T.nnet.sigmoid(X.dot(self.W)+self.bias_h)

    def output(self, X):
        Z = self.forward_hidden(X)
        return T.nnet.sigmoid(Z.dot(self.W.T)+self.bias_o)
        
        
def main():
    Xtr, Ytr, Xts, Yts = util.getKaggleMNIST()
    autoencoder = Autoencoder(300,0)
    autoencoder.fit(Xtr,epochs=2)
    done = False

    while not done:
        i = np.random.choice(len(Xts))
        x = Xts[i]
        y = autoencoer.predict([x])
        plt.subplot(1,2,1)
        plt.imshow(x.reshape(28,28), cmap='gray')
        plt.title('Original')
        plt.subplot(1,2,2)
        plt.imshow(y.reshape(28,28), cmap='gray')
        plt.title('Reconstructed')
        plt.show()

        other = input("Generate another? ")
        if 'n' in other.lower().strip() or 'n' in other.lower().strip()[0]:
            done = True
        
        
if __name__ == '__main__':
    main()
