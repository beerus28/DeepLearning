import numpy as np
import os
from loader import loaddata
import matplotlib.pyplot as plt
import copy

def sigm(l):
    return 1./(1. + np.exp(-l))

class DBM:
    def __init__(self, shape):
        self.nvis = shape[0] ## support 2 hidden layer so the shape is [nvis,nh1,nh2]
        self.nh1 = shape[1]
        self.nh2 = shape[2] 
        self.training_data = []
        self.validation_data = []
        self.load()

        ## initialize weights and biases
        np.random.seed()
        self.w1 = np.random.randn(shape[0], shape[1])
        self.w2 = np.random.randn(shape[1], shape[2])

        self.bv = np.zeros(shape[0])
        self.bh1 = np.zeros(shape[1])
        self.bh2 = np.zeros(shape[2])

    def train(self, rate = 0.01, nepoches = 30, batchsize = 30, M=100):
        training_error = []
        validation_error = []
        
        ## iterate epoches
        for epoch in xrange(nepoches):
            
            ## shuffle and divide training dataset into small batches for stochastic gradient descent
            batches = self.reshuffle(batchsize)
     
            ## iterate batch
            for batch in batches:

                ## Variational Inference:
                mu1 = np.random.rand(batchsize, self.nh1)
                mu2 = np.random.rand(batchsize, self.nh2)

                for i in xrange(10):
                    for n in xrange(batchsize):
                        mu1[n] = sigm(np.dot(self.w1.T, batch[n]) + np.dot(self.w2,mu2[n]) + self.bh2)
                        mu2[n] = sigm(np.dot(self.w2.T, mu1[n]) + self.bh2)

                ## Stochastic Approximation, Persistent CD:
                samplev = np.random.RandomState().binomial(n=1, p=0.5, size=(M, self.nvis))
                sampleh1 = np.random.RandomState().binomial(n=1, p=0.5, size=(M, self.nh1))
                sampleh2 = np.random.RandomState().binomial(n=1, p=0.5, size=(M, self.nh2))

                for i in xrange(M):
                    sampleh1[i], sampleh2[i], samplev[i] = self.samples(sampleh1[i], sampleh2[i], samplev[i])

                ## Update parameters
                self.w1 += rate * (np.dot(batch.T, mu1)/float(batchsize) -np.dot(samplev.T, sampleh1)/float(M))
                self.w2 += rate * (np.dot(mu2.T,mu1)/float(batchsize) - np.dot(sampleh1.T, sampleh2)/float(M))
                
            ## calculate training error and validation error
            ce_errorrate = self.get_cross_entropy(self.training_data)
            vce_errorrate = self.get_cross_entropy(self.validation_data)
            training_error.append(ce_errorrate)
            validation_error.append(vce_errorrate)
            print ce_errorrate
        return training_error, validation_error

                
    def get_cross_entropy(self, pos_v_sample): 
        error = 0.
        for n in range(pos_v_sample.shape[0]):
            v = pos_v_sample[n, :]
            h1 = np.random.randint(2, size=(self.nh1, ))
            h2 = np.random.randint(2, size=(self.nh2, ))

            # sample h1
            possibility_h1 = sigm(np.dot(self.w1.T, v) + np.dot(self.w2, h2) + self.bh1)
            h1 = self.binarize(possibility_h1)
            
            # sample h2 
            possibility_h2 = sigm(np.dot(self.w2.T, h1) + self.bh2)
            h2 = self.binarize(possibility_h2)
            
            # sample V 
            possibility_v = sigm(np.dot(self.w1, h1) + self.bv)
        
            error -= (np.dot(v, np.log(possibility_v)) + np.dot(1.0 - v, np.log(1.0 - possibility_v)))
        return error / pos_v_sample.shape[0]
    
    def binarize(self, possibility):
        rand = np.random.rand(possibility.shape[0])
        l = np.zeros(possibility.shape[0])
        l[possibility > rand] = 1.0
        return l
    
    def samples(self, h1, h2, v):
        # sample h1
        possibility_h1 = sigm(np.dot(self.w1.T, v) + np.dot(self.w2, h2) + self.bh1)
        h1 = self.binarize(possibility_h1)
        
        # sample h2 
        possibility_h2 = sigm(np.dot(self.w2.T, h1) + self.bh2)
        h2 = self.binarize(possibility_h2)
        
        # sample V 
        possibility_v = sigm(np.dot(self.w1, h1) + self.bv)
        v = self.binarize(possibility_v)

        return h1, h2, v
        
    def reshuffle(self,batchsize):
        np.random.shuffle(self.training_data)
        batches = []
        for i in xrange(0, len(self.training_data), batchsize):
            batches = batches + [self.training_data[i:i+batchsize]]
        return batches

    def load(self): #load txt files (train, validate, test) into memory
        self.training_data, self.validation_data, self.testing_data = loaddata();
        return


if __name__ == "__main__":
    dbm = DBM([784,100,100])
    te,ve = dbm.train() # adjust training parameters in function

    ## plot cross-entropy errors
    plt.ylabel('Cross-entropy Error')
    plt.xlabel('Epoch')
    x = range(len(te))
    plt.plot(x,te, 'r', label = "training")
    plt.plot(x,ve, 'b', label = "validation")
    plt.legend()
    plt.show()

    ## plot weights
    np.savetxt('W1.csv', dbm.w1, delimiter=',')
    np.savetxt('W2.csv', dbm.w1, delimiter=',')

    ## 100 gibbs sampling chains
    samplev = np.random.RandomState().binomial(n=1, p=0.5, size=(100, dbm.nvis))
    sampleh1 = np.random.RandomState().binomial(n=1, p=0.5, size=(100, dbm.nh1))
    sampleh2 = np.random.RandomState().binomial(n=1, p=0.5, size=(100, dbm.nh2))

    for i in xrange(100): # 100 sample chains
        for j in xrange(1000): # 1000 gibss sampling step
            sampleh1[i], sampleh2[i], samplev[i] = dbm.samples(sampleh1[i], sampleh2[i], samplev[i])

    np.savetxt('samples.csv', samplev, delimiter=',')

    
    



    
            
        


    
        

        


                
        
        
