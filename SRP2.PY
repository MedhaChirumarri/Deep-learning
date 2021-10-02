import pandas as pd 
import numpy as np
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import time



folders = glob.glob("/home/system/Desktop/SRP/train/*")
Xtrain=[]
Ytrain=[]
k=0
for folder in folders:
    for filename in glob.glob(folder+"/*.csv"):
        df = pd.read_csv(filename, index_col=None, header=0,encoding='utf-16')
        Ytrain.append(np.array(k))
        Xtrain.append(np.array(df))
    k=k+1

print(len(Xtrain))
folders = glob.glob("/home/system/Desktop/SRP/test/*")
Xtest=[]
Ytest=[]
k=0
check_p=0
m=0
for folder in folders:
    for filename in glob.glob(folder+"/*.csv"):
        df = pd.read_csv(filename, index_col=None, header=0,encoding='utf-16')
        Ytest.append(np.array(k))
        Xtest.append(np.array(df))
    k=k+1

print(len(Xtest))

def eval(net,criterion,batch_size,topk, X, Y):
    correct = 0
    files,langs = batched_dataloader(batch_size, X, Y,False)
    hidden = net.init_hidden(batch_size)
    output, hidden = net(files, hidden)
    val, indices = output.topk(topk)
    for i,j in zip(indices,langs):
        if j in i:
            correct += 1
    accuracy = correct/batch_size
    loss=criterion(output, langs)
    return accuracy,loss

def batched_dataloader(batch_size, X, Y,train=True):
    files = []
    langs = []
    file_len=[]
    if train==True:
        for i in range(batch_size):
            index = np.random.randint(len(X))
            file, lang = X[index], Y[index]
            file_len.append(len(file))
            files.append(torch.tensor(file))
            langs.append(lang)
        
    else:
        langs=Y
        file_len=[len(i) for i in X]
        files=[torch.tensor(i) for i in X]
    
    langs=np.array(langs)
    files=nn.utils.rnn.pad_sequence(files, batch_first=True)
    files=np.array(files)
    
    
    k=np.empty([max(file_len),batch_size,80])
    for i in range(max(file_len)):
        for j in range(batch_size):
            k[i][j]=np.array(files[j][i])
            
    files=torch.from_numpy(k).float()
    langs=torch.from_numpy(langs).long()
    files = nn.utils.rnn.pack_padded_sequence(files,file_len, enforce_sorted = False)
    
    return files, langs


def train_batch(net, opt, criterion, batch_size):
    
    opt.zero_grad()
    
    files, langs = batched_dataloader(batch_size, Xtrain, Ytrain)
    
    hidden = net.init_hidden(batch_size)
    output, hidden = net(files,hidden)
    loss = criterion(output, langs)
    
    loss.backward()
    opt.step()
    print("loss:",loss)
    print(" ")
    return loss



def train_setup(net, lr = 0.01, n_batches = 100, batch_size = 10, momentum = 0.9):
    criterion = nn.NLLLoss()
    opt = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    
    loss_arr = np.zeros(n_batches + 1)
    
    for i in range(n_batches):
        print("batch:",i+1)
        loss_arr[i+1] = (loss_arr[i]*i + train_batch(net, opt, criterion, batch_size))/(i + 1)
    print("avg_loss_array:",loss_arr)
    print(" ")
    print('Top-3:', eval(net, criterion,len(Xtest), 3, Xtest, Ytest), 'Top-2:', eval(net, criterion,len(Xtest), 2, Xtest, Ytest))
                      

class LSTM_net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM_net, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTM(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, input, hidden):
        out, hidden = self.lstm_cell(input, hidden)
        output = self.h2o(hidden[0].view(-1, self.hidden_size))
        output = self.softmax(output)
        return output, hidden
    
    def init_hidden(self, batch_size = 1):
        return (torch.zeros(1, batch_size, self.hidden_size), torch.zeros(1, batch_size, self.hidden_size))
    
    
n_hidden = 128
tic = time.time()
net = LSTM_net(80, n_hidden, 9)
train_setup(net, lr=1, n_batches=20, batch_size = 128)
toc = time.time()
print(" ")
print('Time taken', toc - tic)