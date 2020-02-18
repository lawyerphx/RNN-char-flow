import torch as tf
import numpy as np
from torch.autograd import Variable

'''
cuda_gpu = tf.cuda.is_available()
if(cuda_gpu):
    net = tf.nn.DataParallel(net).cuda()
'''

class RNN(tf.nn.Module):
    def __init__(self, x_size, h_size, y_size):
        super(RNN, self).__init__()
        self.trans_h = tf.nn.Linear(h_size, h_size)
        self.trans_x = tf.nn.Linear(x_size, h_size)
        self.trans_y = tf.nn.Linear(h_size, y_size)
        self.h = tf.randn(1, h_size).requires_grad_(True)

    def forward(self,x):   #make one step in RNN
        y = []
        for i in range(len(x)):
            xt = tf.tensor(x[i].reshape(1,-1)).float()
            self.h = tf.tanh(self.trans_h(self.h) + self.trans_x(xt))
            yt = tf.softmax(self.trans_y(self.h) + 1e-6, dim=0)
            y.append(yt)
        y = tf.cat(y, dim=0)
        return y

#read the data, and caculate the vocabulary size
data = open('input.txt','r').read()
chars = list(set(data))
data_size, voca_size = len(data), len(chars)

#convert between char and index
char_to_index = {ch:i for i,ch in enumerate(chars)}
index_to_char = {i:ch for i,ch in enumerate(chars)}

rnn = RNN(voca_size, 100, voca_size)
#print(rnn)

#build the train data
X_train, y_train = [], []
for i in range(data_size-1):
    v_i = np.zeros(voca_size, dtype=float)
    v_i[ char_to_index[data[i]] ] += 1
    X_train.append(v_i)
    v_it = np.zeros(voca_size, dtype=float)
    v_it[ char_to_index[data[i+1]] ] += 1
    y_train.append(v_it)

X_train = np.array(X_train)
y_train = np.array(y_train)

params = rnn.parameters()

#train the modal
optmizer = tf.optim.SGD(params, lr=0.1)
loss_func = tf.nn.MSELoss()

batch_size = 5

for i in range(len(X_train)):
    mask = np.random.choice(len(X_train), batch_size)
    pred = rnn(X_train[mask])
    aim = tf.from_numpy(y_train[mask]).float()
    print (pred.shape, aim.shape)
    loss = loss_func(pred, aim)
    optmizer.zero_grad()
    loss.backward()
    optmizer.step()
