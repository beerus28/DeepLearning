# preactlayer[] -- pre-activation layer before sigmoid
# postactlayer[] -- layer to store post-activation results after sigmoid 
# outputlayer[] -- output layer which applies softmax to the last postactlayer


import numpy as np
import os
import copy
from loader import loaddata
import matplotlib.pyplot as plt

def sigmoid(l):
    return 1.0/(1.0 + np.exp(-l))

def sigmoid_deriv(l):
    return sigmoid(l) * (1 - sigmoid(l))

def softmax(l): # prerequisite l has to be a list
    return np.exp(l)/np.sum(np.exp(l))

def softmax_deriv(l):
    return softmax(l) * (1 - softmax(l))

def operate_on_Narray(A, B, function):
    return [function(a,b) for a, b in zip(A, B)]


class A1_NeuralNetwork(object):

    def __init__(self):
        self.arch = [] #architecture
        self.num_layers = 0 #len(architecture)

        self.weights = []
        self.biases = []
        self.preactlayer = []
        self.postactlayer = []
        self.outputlayer = []
        
        ## predeclared training, validation, and test datasets
        ## keep weight, bias, preactlayer, and postactlayer to have the same number of layers for simplicity
        self.training_data = ()
        self.validation_data = ()
        self.testing_data = ()

        self.load()

    def load(self): #load txt files (train, validate, test) into memory
        self.training_data, self.validation_data, self.testing_data = loaddata();
        return
    
    def train(self, numlayer = 1, numnode = 100, rate = 0.6, num_epoches = 30, momentum = 0.9, batchsize = 10): #train the neural network with datasets preloaded into memory
        self.num_layers = numlayer + 2
        self.arch = [784]
        for i in xrange(numlayer):
            self.arch.append(numnode)
        self.arch.append(10)

        

        # shuffle and divide training dataset into small batches for stochastic gradient descent
        np.random.shuffle(self.training_data)
        batches = [] 
        for i in xrange(0,len(self.training_data), batchsize):
            batches = batches  +  [self.training_data[i:i+batchsize]]
        
        ## initialize weight
        np.random.seed()

        self.weights = [np.zeros((1,1))] 
        for r, c in zip(self.arch[1:], self.arch[:-1]):
            self.weights = self.weights + [((6**0.5)/(19**0.5))*np.random.randn(r,c)]
            
        ## initialize bias, preactive and postactive layers
        for i in self.arch:
            self.biases = self.biases + [np.zeros((i,1))]
            self.preactlayer = self.preactlayer + [np.zeros((i,1))]
            self.postactlayer = self.postactlayer + [np.zeros((i,1))]


        training_error = []
        validation_error = []

        ## epoch
        for epoch in xrange(num_epoches):
            prev_deriv_b = [np.zeros(b.shape) for b in self.biases]
            prev_deriv_w = [np.zeros(w.shape) for w in self.weights]

            each_classification_successcount = 0 # for plot
            each_cross_entropy_error = 0
            inst_count = 0
            
            for batch in batches:
                deriv_b = [np.zeros(b.shape) for b in self.biases]
                deriv_w = [np.zeros(w.shape) for w in self.weights]

                #accumulate derivatives of w and b from the current batch of data
                for instance in batch:
                    error1, error2 = self.fprop(instance)
                    each_cross_entropy_error = each_cross_entropy_error + error1
                    each_classification_successcount = each_classification_successcount + error2
                    inst_count = inst_count + 1
                    unit_deriv_w, unit_deriv_b = self.bprop(instance)

##                    for i in xrange(len(self.biases)):
##                        for j in xrange(len(self.biases[i])):
##                            deriv_b[i][j] = deriv_b[i][j] + delta_deriv_b[i][j]

                    # optimized faster matrix manipulation
                    deriv_b = operate_on_Narray(deriv_b, unit_deriv_b, lambda a, b: a+b)
                    deriv_w = operate_on_Narray(deriv_w, unit_deriv_w, lambda a, b: a+b)

                # add momentum

                deriv_b = operate_on_Narray(prev_deriv_b, deriv_b, lambda a,b: momentum*a + (rate/batchsize)*b)
                deriv_w = operate_on_Narray(prev_deriv_w, deriv_w, lambda a,b: momentum*a + (rate/batchsize)*b)

                prev_deriv_b = np.copy(deriv_b);
                prev_deriv_w = np.copy(deriv_w);


                #update weights and biases based on the averaged derivatives computed from one btch of data
                self.biases = operate_on_Narray(self.biases, deriv_b, lambda a, b: a-b)
                self.weights = operate_on_Narray(self.weights, deriv_w, lambda a, b: a-b)

            # stats on current epoch
            ce_errorrate = 1.0*each_cross_entropy_error/inst_count
            classification_rate = 100.0 - 100.0*each_classification_successcount/inst_count
            vce_errorrate,vclassification_rate = self.test()

            training_error.append((ce_errorrate, classification_rate))
            validation_error.append((vce_errorrate,vclassification_rate))


            #print("Round {0}, cross-entropy error {1} training: error rate {2},".format(epoch + 1,ce_errorrate,classification_rate ))
            #print("Round {0}, cross-entropy error {1}, validation: error rate {2}.".format(epoch + 1, vce_errorrate, vclassification_rate))

        return training_error, validation_error

    def predict(self, instance): #predict one instance
        return self.fprop(instance, True)
        

    def test(self): #test the accuracy with datasets preloaded into memory
        results = [(self.predict(instance)) for instance in self.validation_data]
        return sum(result[0] for result in results)/len(results), 100.0 - 100.0*sum(result[1] for result in results)/len(results)

    def fprop(self, inst, validate = False): #forward propagation, which updates both pre- and post-active layers, it also returns the predicted results
        x = inst[0]
        self.postactlayer[0] = x

        for i in xrange(1, self.num_layers):
            self.preactlayer[i] = (self.weights[i].dot(self.postactlayer[i-1]) + self.biases[i])
            self.postactlayer[i] = sigmoid(self.preactlayer[i])

        self.outputlayer = softmax(self.postactlayer[-1])

        cross_entropy_error = -self.outputlayer.transpose().dot(inst[1])[0][0]

        return cross_entropy_error, np.argmax(self.outputlayer) == np.argmax(inst[1]) ## validation set has different result format


    def bprop(self,inst): #backward propagation, which returns nabla weights and nabla bias
                          #waiting to be accumulated and will eventually be used to update weights and biases
        y = inst[1]

        deriv_b = [np.zeros(bias.shape) for bias in self.biases]
        deriv_w = [np.zeros(weight.shape) for weight in self.weights]

        cross_entropy_loss = (self.outputlayer - y) * sigmoid_deriv(self.preactlayer[-1])
        
        deriv_b[self.num_layers-1] = cross_entropy_loss
        deriv_w[self.num_layers-1] = np.dot(cross_entropy_loss, self.postactlayer[-2].transpose())

        for l in xrange(self.num_layers - 3, -1, -1):
            cross_entropy_loss = np.multiply(
                self.weights[l + 2].transpose().dot(cross_entropy_loss),
                sigmoid_deriv(self.preactlayer[l + 1])
            )
            deriv_b[l + 1] = cross_entropy_loss
            deriv_w[l + 1] = np.dot(cross_entropy_loss,self.postactlayer[l].transpose())

        return deriv_w, deriv_b
    def clear(self):
        self.weights = []
        self.biases = []
        self.preactlayer = []
        self.postactlayer = []
        self.outputlayer = []
        return


def main():

    numround = 7
    
    nn = A1_NeuralNetwork()

    plot_training_ceerror = []
    plot_training_classification = []

    plot_validation_ceerror = []
    plot_validation_classification = []

    # first round for initialization
    training_stats, validation_stats = nn.train()

    x = range(len(training_stats))

    for e in training_stats:
        plot_training_ceerror.append(e[0])
        plot_training_classification.append(e[1])

    for e in validation_stats:
        plot_validation_ceerror.append(e[0])
        plot_validation_classification.append(e[1])

    # four more rounds
    for i in xrange(numround-1):
        nn.clear()
        training_stats, validation_stats = nn.train()
        for l in xrange(len(training_stats)):
            plot_training_ceerror[l] = plot_training_ceerror[l] + training_stats[l][0]
            plot_training_classification[l] = plot_training_classification[l] + training_stats[l][1]

            plot_validation_ceerror[l] = plot_validation_ceerror[l] + validation_stats[l][0]
            plot_validation_classification[l] = plot_validation_classification[l] + validation_stats[l][1]

    for i in xrange(len(training_stats)):
        plot_training_ceerror[i] = plot_training_ceerror[i]/numround
        plot_training_classification[i] = plot_training_classification[i]/numround
        plot_validation_ceerror[i] = plot_validation_ceerror[i]/numround
        plot_validation_classification[i] = plot_validation_classification[i]/numround        
    
    #plots
    plt.ylabel('Cross-entropy Error')
    plt.xlabel('Epoch')
    plt.plot(x,plot_training_ceerror, 'r', label = "training")
    plt.plot(x,plot_validation_ceerror, 'b', label = "validation")
    plt.legend()
    plt.show()

    plt.ylabel('Classification Error')
    plt.xlabel('Epoch')
    plt.plot(x,plot_training_classification, 'r', label = "training")
    plt.plot(x,plot_validation_classification, 'b', label = "validation")
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
                image[r][c] = image[r][c] + nn.weights[1][i][r*28 + c]

    plt.imshow(image,cmap=plt.get_cmap('gray'), interpolation='none')
    plt.show()


if __name__ == "__main__":
    main()



    







