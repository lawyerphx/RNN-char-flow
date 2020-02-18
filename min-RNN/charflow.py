import torch as tf

'''
def FC_layer(x, out_size, activation_func=None):
    input_dim = x.shape[1]
    W = tf.rand(input_dim, out_size)
    b = tf.rand(1,out_size)
    y = tf.dot(x, W) + b
    if activation_func != None: y = activation_func(y)
    return y
'''

cuda_gpu = torch.cuda.is_available()
if(cuda_gpu):
    net = torch.nn.DataParallel(net).cuda()

class RNN:
    def __init__(self,input_size, hidden_size, output_size):
        self.h = tf.rand(hidden_size)
        self.bh = tf.zeros(hidden_size)
        self.by = tf.zeros(output_size)
        self.Whh = tf.rand(hidden_size, hidden_size)
        self.Why = tf.rand(output_size, hidden_size)
        self.Wxh = tf.rand(hidden_size, input_size)

    def loss(self,x,y):
        return tf.mean(tf.square(self.step(x)-y))

    def step(self,x):   #make one step in RNN
        self.h = tf.tanh(tf.dot(self.Whh, self.h) + tf.dot(self.Wxh, x)) + self.bh
        y = tf.softmax(tf.dot(self.Why, self.h) + self.by + 1e-6, dim=0)
        return y

    def train(self,X,y):    #train the model on X,y
        com_loss = 0
        for i in range(len(X)):
            com_loss += self.loss(X[i],y[i])
        com_loss /= len(X)

#read the data, and caculate the vocabulary size
data = open('input.txt','r').read()
chars = list(set(data))
data_size, voca_size = len(data), len(chars)
print ('data has %d characters, %d unique.' % (data_size, voca_size))

#convert between char and index
char_to_index = {ch:i for i,ch in enumerate(chars)}
index_to_char = {i:ch for i,ch in enumerate(chars)}

rnn = RNN(voca_size, 100, voca_size)

#build the train data
X_train, y_train = [], []
for i in range(data_size-1):
    v_i = tf.zeros(voca_size)
    v_i[ char_to_index[data[i]] ] += 1
    X_train.append(v_i)
    v_it = tf.zeros(voca_size)
    v_it[ char_to_index[data[i+1]] ] += 1
    y_train.append(v_it)

#train the modal
rnn.train(X_train,y_train)

