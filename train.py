import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import numpy as np
from preprocessing import tokenize,stem,bag_of_words
with open('data.json','r') as f:
    data  = json.load(f)
    
all_words = []
tags = []
xy = []

for dat in data['data']:
    tag = dat['tag']
    tags.append(tag)
    for pattern in dat['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w,tag))

ignore = ["?",".","!",","]
all_words = sorted(set([stem(w) for w in all_words if w not in ignore]))
tags = sorted(set(tags))
X_train = []
Y_train = []
for sentence,tag in xy:
    bag = bag_of_words(sentence,all_words)
    label = tags.index(tag)
    X_train.append(bag)
    Y_train.append(label)

X_train = torch.stack(X_train)
Y_train = torch.tensor(Y_train)

class ChatDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train
    def __len__(self):
        return self.n_samples
    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]

batch_size = 8 
dataset = ChatDataset()
train_data = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)