import pandas as pd 
import numpy as np
import glob
import torch
import torch.nn as nn
import torch.optim as optim



folders = glob.glob("/home/system/Desktop/SRP/test/*")
X=[]
Y=[]
k=0
for folder in folders:
    for filename in glob.glob(folder+"/*.csv"):
        df = pd.read_csv(filename, index_col=None, header=0,encoding='utf-16')
        Y.append(np.array(k))
        X.append(np.array(df))
    k=k+1



def batched_dataloader(batch_size, X, Y):
    files = []
    langs = []
    file_len=[]
    for i in range(batch_size):
        index = np.random.randint(len(X))
        file, lang = X[index], Y[index]
        file_len.append(len(file))
        files.append(torch.tensor(file))
        langs.append(lang)
    langs=np.array(langs)
    files=nn.utils.rnn.pad_sequence(files, batch_first=True)
    files=np.array(files)
    files=torch.from_numpy(files).float()
    langs=torch.from_numpy(langs).long()
    return files, langs


def train(net, opt, criterion, batch_size):
    
    opt.zero_grad()
    
    files, langs = batched_dataloader(batch_size, X, Y)
    
    hidden = net.init_hidden(batch_size)
    output, hidden = net(files,hidden)
    loss = criterion(output, langs)
    
    loss.backward()
    opt.step()
    print("loss:",loss)
    return loss



def train_setup(net, lr = 0.01, n_batches = 100, batch_size = 10, momentum = 0.9):
    criterion = nn.NLLLoss()
    opt = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    
    loss_arr = np.zeros(n_batches + 1)
    
    for i in range(n_batches):
        print(i)
        loss_arr[i+1] = (loss_arr[i]*i + train(net, opt, criterion, batch_size))/(i + 1)
    print(loss_arr)
            
    #print('Top-1:', eval(net, len(X), 1, X, Y), 'Top-2:', eval(net, len(X), 2, X, Y))
                      

class LSTM_net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM_net, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTM(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, input, hidden):
        out, hidden = self.lstm_cell(input.view(input.size()[1],64,80), hidden)
        output = self.h2o(hidden[0].view(-1, self.hidden_size))
        output = self.softmax(output)
        return output.view(64,9), hidden
    
    def init_hidden(self, batch_size = 1):
        return (torch.zeros(1, batch_size, self.hidden_size), torch.zeros(1, batch_size, self.hidden_size))
    
    
n_hidden = 128
net = LSTM_net(80, n_hidden, 9)
train_setup(net, lr=0.15, n_batches=50, batch_size = 64)
