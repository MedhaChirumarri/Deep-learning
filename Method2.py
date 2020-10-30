#method 2

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def accuracy(y_hat, y):
  pred = torch.argmax(y_hat, dim=1)
  return (pred == y).float().mean()

data1=pd.read_csv("/hin13.csv",encoding="utf-16")
data2=pd.read_csv("/kan12.csv",encoding="utf-16")
data1["class"]=[1 for i in range(len(data1))]
data2["class"]=[2 for i in range(len(data2))]
data=data1.append(data2)

Y=np.asarray(data["class"])
X=np.asarray(data.drop("class",axis=1))

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)
X_train, Y_train, X_test, Y_test = map(torch.tensor, (X_train, Y_train, X_test, Y_test))
X_train=X_train.float()
def fit_v1(epochs = 1000, learning_rate = 1):
  loss_arr = []
  acc_arr = []
  opt = optim.SGD(fn.parameters(), lr=learning_rate)
  
  for epoch in range(epochs):
    y_hat = fn(X_train)
    loss = F.cross_entropy(y_hat, Y_train)
    loss_arr.append(loss.item())
    acc_arr.append(accuracy(y_hat, Y_train))

    loss.backward()
    opt.step()
    opt.zero_grad()
        
  plt.plot(loss_arr, 'r-')
  plt.plot(acc_arr, 'b-')
  plt.show()      
  print('Loss before training', loss_arr[0])
  print('Loss after training', loss_arr[-1])
  
class FirstNetwork_v1(nn.Module):
  
  def __init__(self):
    super().__init__()
    torch.manual_seed(0)
    self.lin1 = nn.Linear(80, 700)
    self.lin2 = nn.Linear(700, 500)
    self.lin3 = nn.Linear(500,5)
    
  def forward(self, X):
    a1 = self.lin1(X)
    h1 = a1.sigmoid()
    a2 = self.lin2(h1)
    h2= a2.sigmoid()
    a3 = self.lin3(h2)
    h3 = a3.exp()/a3.exp().sum(-1).unsqueeze(-1)
    return h3

fn = FirstNetwork_v1()
fit_v1()

fn = FirstNetwork_v1()
fit()
