The code requires both RBM_MNIST.py and loader.py in a same folder. Please run the RBM_MNIST.py to train the RBM and the plots will be shown when the code finishes running. The 100 samples for problem 5 c) will first show, followed by cross-entropy error plot and the visualization of learned weights.

To change parameters, such as number of epoches, k value, etc., please change the arguments passed to the rbm.train()

Below is the declaration of train method: 

def train(self, rate = 0.01, nepoches = 20, k = 1, batchsize = 30):

with each parameter representing:

rate -- learning rate
nepoches -- number of epoches
k -- k values in gibbs sampling
batchsize -- mini batch size for gradient descent updates

To change the number of rounds we want to train the neural network for an averaged result, we can change the variable "numround" in the main function.

For problem 5d), I reused some code froim assignment 1. The RBM_MNIST.py will save the pretrained weights into a csv file called "weights.csv". In the "assignment1 code" folder, we cal simply run mnist.py to load this csv file for initializing weights. The results are ploted at the end.


