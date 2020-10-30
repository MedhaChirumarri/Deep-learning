import pandas as pd 
import numpy as np
import math
import glob
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.model_selection import train_test_split


def f(path,x):
    
    all_files = glob.glob(path + "/*.csv")
    
    li = []
    
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0,encoding='utf-16')
        li.append(df)
    
    frame = pd.concat(li, axis=0, ignore_index=True)
    frame["class"]=[x for i in range(len(frame))]
    return (frame)

data1=f("/content/kan.zip",0)
data2=f("/content/hin.zip",1)
data3=f("/content/asm.zip",2)
data4=f("/content/ben.zip",0)
data5=f("/content/guj",2)
data1=data1.append(data2)
data=data3.append(data1)
data6=data4.append(data5)
Y=np.asarray(data["class"])
X=np.asarray(data.drop("class",axis=1))
Y1=np.asarray(data6["class"])
X1=np.asarray(data6.drop("class",axis=1))
#print(type(X))
#X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)
#X_train, Y_train, X_test, Y_test = map(torch.tensor, (X_train, Y_train, X_test, Y_test))
X,Y=map(torch.tensor,(X,Y))
X1,Y1=map(torch.tensor,(X1,Y1))

def accuracy(y_hat, y):
  pred = torch.argmax(y_hat, dim=1)
  return (pred == y).float().mean()


def fit(epochs = 1000, learning_rate = 1):
  loss_arr = []
  acc_arr = []
  for epoch in range(epochs):
   # y_hat = fn(X_train)
   # loss = F.cross_entropy(y_hat, Y_train)
    #loss_arr.append(loss.item())
    #acc_arr.append(accuracy(y_hat, Y_train))
    #y_hat = fn(X_train)
    y_hat = fn(X)
    
    loss = F.cross_entropy(y_hat, Y)
    loss_arr.append(loss.item())
    acc_arr.append(accuracy(y_hat, Y))

    loss.backward()
    with torch.no_grad():
      for param in fn.parameters():
        param -= learning_rate * param.grad
      fn.zero_grad()
        
  plt.plot(loss_arr, 'r-')
  plt.plot(acc_arr, 'b-')
  plt.show()      
  print('acurracy', acc_arr[-1])
  print('Loss after training', loss_arr[-1])

class FirstNetwork(nn.Module):
  
  def __init__(self):
    super().__init__()
    torch.manual_seed(0)
    self.weights1 = nn.Parameter(torch.randn(80,700) / math.sqrt(2))
    self.bias1 = nn.Parameter(torch.zeros(700))
    self.weights2 = nn.Parameter(torch.randn(700, 500) / math.sqrt(2))
    self.bias2 = nn.Parameter(torch.zeros(500))
    self.weights3 = nn.Parameter(torch.randn(500, 5) / math.sqrt(2))
    self.bias3 = nn.Parameter(torch.zeros(5))
    
  def forward(self, X):
    a1 = torch.matmul(X, self.weights1.double()) + self.bias1
    h1 = a1.sigmoid()
    a2 = torch.matmul(h1, self.weights2.double()) + self.bias2
    h2 = a2.sigmoid()
    a3=torch.matmul(h2, self.weights3.double()) + self.bias3
    h3 = a3.exp()/a3.exp().sum(-1).unsqueeze(-1)
    return h3

fn = FirstNetwork()
fit()

y1=fn(X1)
acc=accuracy(y1,Y1)
print(acc)

