import numpy as np
import pandas as pd
from sklearn.utils import shuffle

def relu(x):
    return x * (x>0)

def error(p,t):
    return np.mean(p!=t)

def getKaggleMNIST():
    df = pd.read_csv("Dataset/train.csv").as_matrix().astype(np.float32)
    df = shuffle(df) # just sort the matrix in random way.
    
    xLabel_train = df[:-1000,1:]/255.0 #dataset has shape 42000x785
                                     #each row is an image
    yLabel_train = df[:-1000,0]


    xLabel_test = df[-1000,1:] / 255.0
    yLabel_test = df[-1000,0]

    return xLabel_train, yLabel_train, xLabel_test, yLabel_test

def init_weights(W):
    return np.random.rand(*W)/ np.sqrt(sum(W))

