from sklearn.utils import class_weight
from scipy import interpolate
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import roc_curve
import numpy as np
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import torch
from torch import Tensor
import torch.nn as nn
# from _internally_replaced_utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset, Subset
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torch.optim as optim
import torchvision.models as models
import torch.distributed as dist
import sklearn
from sklearn.metrics import cohen_kappa_score, accuracy_score
import time
import os
from tqdm import tqdm_notebook
from torch.optim.lr_scheduler import StepLR
from model import ResNet18WithAttention,ResNet34WithAttention,ResNet50WithAttention
from sklearn.model_selection import train_test_split



#Choosing device
device = torch.device('cuda' if torch.cuda.is_available()
                      else 'cpu')  
print("Number of GPUs:", torch.cuda.device_count())


for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
print(device)
print(torch.version.cuda)
data = pd.read_csv('/gpfs/scratch/jy4684/ondemand/dl/train.csv')
print('Train Size = {}'.format(len(data)))
print(data.head()) 

counts = data['diagnosis'].value_counts()
class_list = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferate']
for i, x in enumerate(class_list):
    counts[x] = counts.pop(i)
class_weights = class_weight.compute_class_weight(
    class_weight='balanced', classes=np.array([0, 1, 2, 3, 4]), y=data['diagnosis'].values)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print(class_weights)


class dataset(Dataset):
    
    def __init__(self, df, data_path, image_transform=None, train=True):
        super(Dataset, self).__init__()
        self.df = df
        self.data_path = data_path
        self.image_transform = image_transform
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_id = self.df['id_code'][index]
        image = Image.open(f'{self.data_path}/{image_id}.png')
        if self.image_transform:
            image = self.image_transform(image)

        if self.train:
            label = self.df['diagnosis'][index]
            return image, label

        else:
            return image
        

#loading and setting up dataset
image_transform = transforms.Compose([transforms.Resize([224, 224]),
                                      transforms.ToTensor(),
                                      torchvision.transforms.RandomHorizontalFlip(
                                          p=0.5),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])  
data_set = dataset(data, f'/gpfs/scratch/jy4684/ondemand/dl/train_images', image_transform=image_transform)

train_set, valid_set = torch.utils.data.random_split(data_set, [3302, 360])

train_dataloader = DataLoader(train_set, batch_size=4, shuffle=True)
valid_dataloader = DataLoader(valid_set, batch_size=4, shuffle=False)



model = ResNet50WithAttention(num_classes=5)


model = model.to('cuda:0')


#Training stage of the model
def train(dataloader, model, loss_fn, optimizer):
    
    model.train()  

    total = 0
    correct = 0
    running_loss = 0

    
    for x,y in dataloader:
        data, target = x.to('cuda:0'), y.to('cuda:0')
        output = model(data)
        loss = loss_fn(output,target)
        
        running_loss += loss.item()
 
        total += y.size(0)
        predictions = output.argmax(dim=1).cpu().detach()
        correct += (predictions == y.cpu().detach()).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = running_loss/len(dataloader)
    avg_acc = 100*(correct/total)

    print(f'\nTraining Loss per batch = {avg_loss:.6f}', end='\t')
    print(
        f'Accuracy on Training set = {100*(correct/total):.6f}% [{correct}/{total}]')

    return avg_loss, avg_acc

#Validation stage of the model
def validate(dataloader, model, loss_fn):
    
    model.eval()

    total = 0
    correct = 0
    running_loss = 0

    with torch.no_grad():

        for x, y in dataloader:

            output = model(x.to('cuda:0'))
            loss = loss_fn(output, y.to('cuda:0')).item()
            running_loss += loss

            total += y.size(0)
            predictions = output.argmax(dim=1).cpu().detach()
            correct += (predictions == y.cpu().detach()).sum().item()

    avg_loss = running_loss/len(dataloader)
    avg_acc = 100*(correct/total)

    print(f'\nValidation Loss per batch = {avg_loss:.6f}', end='\t')
    print(
        f'Accuracy on Validation set = {100*(correct/total):.6f}% [{correct}/{total}]')
    

    return avg_loss, avg_acc


# Using crossentropy as the loss function and SGD as the optimizer
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=10, gamma=0.8)
nb_epochs = 50


train_losses = []
valid_losses = []
train_acc = []
valid_acc = []

for epoch in range(nb_epochs):
    print(f'\nEpoch {epoch+1}/{nb_epochs}')
    print('-------------------------------')
   
    train_loss, train_a = train(
        train_dataloader, model, loss_fn, optimizer)
    train_losses.append(train_loss)
    train_acc.append(train_a)
    
    valid_loss, valid_a = validate(valid_dataloader, model, loss_fn)
    valid_losses.append(valid_loss)
    valid_acc.append(valid_a)
    scheduler.step()
    torch.save({
        
        'model_state_dict': model.state_dict(),
        
    }, f'/gpfs/scratch/jy4684/ondemand/dl/model_epoch_{epoch}.pth')
    print(f'Model saved for epoch {epoch}')

def save_as(x, filename='my_array'):
    array = np.array(x)
    np.save(f'{filename}.npy', array)
    
train_losses=np.array(train_losses)
valid_losses=np.array(valid_losses)
train_acc=np.array(train_acc)
valid_acc=np.array(valid_acc)

save_as(train_losses,'/gpfs/scratch/jy4684/ondemand/dl/train_losses')
save_as(valid_losses,'/gpfs/scratch/jy4684/ondemand/dl/valid_losses')
save_as(train_acc,'/gpfs/scratch/jy4684/ondemand/dl/train_acc')
save_as(valid_acc,'/gpfs/scratch/jy4684/ondemand/dl/valid_acc')

