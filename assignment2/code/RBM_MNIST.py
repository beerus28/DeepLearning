import numpy as np
import os
from loader import loaddata
import matplotlib.pyplot as plt
import copy
import csv

def sigmoid(l):
    return 1.0/(1.0 + np.exp(-l))

def operate_onNarray(A, B, function):
    return [function(a,b) for a, b in zip(A, B)]


class A2_RestrictedBoltzmannMachine(object):
    def __init__(self, nvisible, nhidden):
        self.nvisible = nvisible
        self.nhidden = nhidden
        self.arch = [nvisible, nhidden]

        self.training_data = []
        self.validation_data = []
        self.testing_data = []

        self.load()

        ## initialize weight
        np.random.seed()
        self.weights = np.random.randn(nvisible,nhidden)

        ## initialize vh and hv biasis
        self.vhbias = np.zeros(nhidden)
        self.hvbias = np.zeros(nvisible)

    def load(self): #load txt files (train, validate, test) into memory
        self.training_data, self.validation_data, self.testing_data = loaddata();
        return

    def train(self, rate = 0.01, nepoches = 20, k = 1, batchsize = 30): #train the RBM with datasets preloaded into memory

        training_error = []
        validation_error = []

        ## iterate epoches
        for epoch in xrange(nepoches):

            # shuffle and divide training dataset into small batches for stochastic gradient descent
            batches = self.reshuffle(batchsize)
            ## iterate batch
            for batch in batches:

                pos_v_sample = batch
                pos_h_sample = self.gibbssample_geth_withv(pos_v_sample)

                neg_v_sample = self.gibbssample_getv_withh(pos_h_sample)
                neg_h_sample = self.gibbssample_geth_withv(pos_v_sample)
                
                ## iterate k
                for step in xrange(1,k):
                    neg_v_sample = self.gibbssample_getv_withh(neg_h_sample)
                    net_h_sample = self.gibbssample_geth_withv(neg_v_sample)

                ## add up k in one batch
                delta_weights = (np.dot(pos_v_sample.T, pos_h_sample) - np.dot(neg_v_sample.T, neg_h_sample))

                delta_hvbias =  np.mean(pos_v_sample - neg_v_sample)
                
                delta_vhbias = np.mean(pos_h_sample - neg_h_sample)


                ## update w, b, and c with averaged gradiant in one batch

                self.weights += (rate) * delta_weights      
                self.hvbias += (rate) * delta_hvbias 
                self.vhbias += (rate) * delta_vhbias

            # Calculate cross entropy error at the end of this epoch

            ce_errorrate = self.get_cross_entropy(self.training_data)
            vce_errorrate = self.get_cross_entropy(self.validation_data)
            training_error.append(ce_errorrate)
            validation_error.append(vce_errorrate)
            
            print("Round {0}, training cross-entropy error {1}, validation cross-entropy error {2}".format(epoch + 1,ce_errorrate,vce_errorrate))

        return training_error, validation_error
                
    def get_cross_entropy(self, pos_v_sample):
        h_sigm = sigmoid(np.dot(pos_v_sample, self.weights) + self.vhbias)
        v_sigm = sigmoid(np.dot(h_sigm, self.weights.T) + self.hvbias)
        cross_entropy =  - np.mean(np.sum(pos_v_sample * np.log(v_sigm) + (1 - pos_v_sample) * np.log(1 - v_sigm), axis=1))
        return cross_entropy
        
    def reshuffle(self,batchsize):
        np.random.shuffle(self.training_data)
        batches = []
        for i in xrange(0, len(self.training_data), batchsize):
            batches = batches + [self.training_data[i:i+batchsize]]
        return batches

    def binarize(self, array): # discrete with binomial distribution
        discretedarray = np.random.RandomState().binomial(n=1,p=array,size=array.shape)
        return discretedarray

    def gibbssample_geth_withv(self, v_sample):
        h_sigm = sigmoid(np.dot(v_sample,self.weights) + self.vhbias)
        h_sample = self.binarize(h_sigm)
        return h_sample

    def gibbssample_getv_withh(self, h_sample):
        v_sigm = sigmoid(np.dot(h_sample, self.weights.T) + self.hvbias)
        v_sample = self.binarize(v_sigm)
        return v_sample
    def sample(self, step):
        np.random.shuffle(self.training_data)
        batch = self.training_data[0:100]
        pos_v_sample = batch
        pos_h_sample = self.gibbssample_geth_withv(pos_v_sample)
        neg_v_sample = self.gibbssample_getv_withh(pos_h_sample)
        neg_h_sample = self.gibbssample_geth_withv(pos_v_sample)
        
        ## iterate k
        for step in xrange(1,step):
            neg_v_sample = self.gibbssample_getv_withh(neg_h_sample)
            net_h_sample = self.gibbssample_geth_withv(neg_v_sample)

        return neg_v_sample
        
        
def main():

    numround = 1
    plot_training_ceerror = []
    plot_validation_ceerror = []

    # first round for initialization
    rbm = A2_RestrictedBoltzmannMachine(784, 100)
    training_stats, validation_stats = rbm.train()
    
    
    x = range(len(training_stats))

    for e in training_stats:
        plot_training_ceerror.append(e)

    for e in validation_stats:
        plot_validation_ceerror.append(e)

    sample = rbm.sample(step = 1000)
    
    #visualize samples
    image = []
    row = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for i in xrange(28):
        image.append(copy.deepcopy(row))

    for i in xrange(100):
        for r in xrange(28):
            for c in xrange(28):
                image[r][c] = image[r][c] + sample[i][r*28 + c]
        plt.subplot(10,10,i)
        plt.imshow(image,cmap=plt.get_cmap('gray'), interpolation='none')
        
        image = []
        for i in xrange(28):
            image.append(copy.deepcopy(row))
        
    plt.show()

    # four more rounds
    for i in xrange(numround-1):
        rbm = A2_RestrictedBoltzmannMachine(784, 100)
        training_stats, validation_stats = rbm.train()
        for l in xrange(len(training_stats)):
            plot_training_ceerror[l] = plot_training_ceerror[l] + training_stats[l]
            plot_validation_ceerror[l] = plot_validation_ceerror[l] + validation_stats[l]


    for i in xrange(len(training_stats)):
        plot_training_ceerror[i] = plot_training_ceerror[i]/numround
        plot_validation_ceerror[i] = plot_validation_ceerror[i]/numround


    #plot cross-entropy errors
    plt.ylabel('Cross-entropy Error')
    plt.xlabel('Epoch')
    plt.plot(x,plot_training_ceerror, 'r', label = "training")
    plt.plot(x,plot_validation_ceerror, 'b', label = "validation")
    plt.legend()
    plt.show()

    #visualize weights
    image = []
    row = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for i in xrange(28):
        image.append(copy.deepcopy(row))

    for i in xrange(100):
        for r in xrange(28):
            for c in xrange(28):
                image[r][c] = image[r][c] + rbm.weights[r*28 + c][i]
        plt.subplot(10,10,i)
        plt.imshow(image,cmap=plt.get_cmap('gray'), interpolation='none')
        
        image = []
        for i in xrange(28):
            image.append(copy.deepcopy(row))
        
    plt.show()

    # save learned weights to csv
    savedWeights = open ('weights.csv', 'wb')
    wr = csv.writer(savedWeights, delimiter=',')
    wr.writerows(rbm.weights.tolist())


if __name__ == "__main__":
    main()
    
