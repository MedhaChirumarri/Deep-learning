import warnings
import pandas as pd
warnings.filterwarnings('ignore')
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch import optim
import numpy as np

data1=pd.read_csv("/content/hin13.csv",encoding="utf-16")
data2=pd.read_csv("/content/kan12.csv",encoding="utf-16")
data1["class"]=[1 for i in range(len(data1))]
data2["class"]=[2 for i in range(len(data2))]
data=data1.append(data2)
Y=np.asarray(data["class"])
X=np.asarray(data.drop("class",axis=1))

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)
X_train, Y_train, X_test, Y_test = map(torch.tensor, (X_train, Y_train, X_test, Y_test))
X_train=X_train.float()


def fit_v2(x, y, model, opt, loss_fn, epochs = 1000):
  
  for epoch in range(epochs):
    loss = loss_fn(model(x), y)

    loss.backward()
    opt.step()
    opt.zero_grad()
    
  return loss.item()


class FirstNetwork_v2(nn.Module):
  
  def __init__(self):
    super().__init__()
    torch.manual_seed(0)
    self.net = nn.Sequential(
        nn.Linear(80, 700).float(), 
        nn.Sigmoid(), 
        nn.Linear(700, 500).float(), 
        nn.Sigmoid(),
        nn.Linear(500,5).float(),
        nn.Softmax()
    )

  def forward(self, X):
    return self.net((X))

fn = FirstNetwork_v2()
loss_fn = F.cross_entropy
opt = optim.SGD(fn.parameters(), lr=1)
print(fit_v2(X_train, Y_train, fn, opt, loss_fn))

fn = FirstNetwork_v2()
fit_v1()
