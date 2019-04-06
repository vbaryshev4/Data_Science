
# coding: utf-8

# In[239]:


#get_ipython().magic('matplotlib inline')
#import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import defaultdict

#from IPython.display import clear_output

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
np.random.seed(0)

from albumentations.pytorch import ToTensor


# In[155]:


from albumentations import (HorizontalFlip,
    IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose, Normalize
)


# In[210]:


albumentations_transform = Compose([
        OneOf([
            IAAAdditiveGaussianNoise(p=0.1),
            GaussNoise(p=0.1),
        ], p=1),
        ShiftScaleRotate(
            shift_limit=0.001, 
            scale_limit=0.001, 
            rotate_limit=13, 
            p=0.5),
#         OneOf([
#             CLAHE(tile_grid_size=(1, 1)),
#             IAASharpen(alpha=(0.9, 0.9)),
#             RandomBrightnessContrast(contrast_limit=1),            
#         ], p=0.5),
        HueSaturationValue(),
        HorizontalFlip(p=0.5),
        Normalize(
            mean=[0.3, 0.4, 0.3],
            std=[0.3, 0.2, 0.3],
        ),
    ToTensor()], p=1)


# In[211]:


from torchvision.datasets import CIFAR10
from PIL import Image
cuda0 = torch.device('cuda:0')
class _CIFAR10(CIFAR10):

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        if self.transform:
            # Apply transformations
            augmented = self.transform(image=img)
            # Convert numpy array to PIL Image
            image = augmented['image']
        
        return image, target


# In[212]:


path = '.'


# In[213]:


train = _CIFAR10(path, train=True, download=True, transform=albumentations_transform)
test = _CIFAR10(path, train=False, download=True, transform=albumentations_transform)


# In[214]:


train[0][0].shape


# In[218]:


#plt.figure(figsize=[6, 6])
#for i in range(4):
#    plt.subplot(2, 2, i + 1)
#    img, label = train[i] 
#    img = img[0]
#    plt.title("Label: {}".format(label))
#    plt.imshow(img, cmap='gray')


# In[222]:


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


# In[247]:


layers = [
    nn.Conv2d(3, 32, kernel_size=3),
    nn.ReLU(),
    nn.Conv2d(32, 32, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Dropout(0.1),
    nn.Conv2d(32, 64, kernel_size=3),
    nn.ReLU(),
    nn.Conv2d(64, 64, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Dropout(0.1),
    Flatten(),
    nn.Linear(1600, 256),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 10)
]


class ConvNet(nn.Module):
    def __init__(self,):
        super(ConvNet, self).__init__()
        self.seq = nn.Sequential(*layers)

    def forward(self, X):
        out = self.seq(X)
        return F.log_softmax(out, dim=-1)


# In[248]:
# cuda0 = torch.device('cuda:0')

train_loader = DataLoader(
    train, 
    batch_size=128, 
    shuffle=True, 
    num_workers=10)


# In[ ]:


#cuda0 = torch.device('cuda:0')
#train_loader.to(cuda0)


# In[253]:


nn_model = ConvNet()
nn_model.to(cuda0)
optimizer = optim.Adam(nn_model.parameters(), lr=0.0005)

losses = []
for epoch in range(1, 250):
    loss_item = 0
    acc = []
    for x, y in tqdm(train_loader):
        optimizer.zero_grad()
        x_cuda = x.to(cuda0)
        y_cuda = y.to(cuda0)
        output = nn_model.forward(x_cuda)
        loss = F.nll_loss(output, y_cuda)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1)[1]
        acc.append(float(pred.eq(y_cuda).sum())/y_cuda.size(0))
        loss_item += loss.item()

    loss_item /= len(train_loader.dataset.targets)
    losses.append(loss_item)
    print('Epoch: ', epoch, 'Loss value: ', loss_item, np.mean(acc))
        

