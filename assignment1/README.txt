The code requires both mnist.py and loader.py in a same folder. Please run the mnist.py to train the neural network and the plots will be shown when the code finishes running. The cross-entropy error plot will first show. After it is closed, then the classification error rate will show. The w will be saved for visualization at the end of minist.py execution.

To change hyper-parameters and neural network structure, please change the arguments passed to nn.train().
Below is the declaration of train method: 

def train(self, numlayer = 1, numnode = 200, rate = 0.1, num_epoches = 200, momentum = 0.5, batchsize = 10):

with each parameter representing:

numlayer -- the numer of hidden layers
numnode -- the number of notes per hidden layer
rate -- learning rate
num_epoches -- number of epoches
momentum -- momentum
batchsize -- mini batch size for gradient descent updates


To change the number of rounds we want to train the neural network for an averaged result, we can change the variable "numround" in the main function.