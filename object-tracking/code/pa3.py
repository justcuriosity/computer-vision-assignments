# -*- coding: utf-8 -*-
"""pa3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Xg-Iqauecoo0Zr3xVIHCgznRWh8GU-fZ
"""

from google.colab import drive
drive.mount('/content/drive')

!ls "/content/drive/My Drive/Colab Notebooks/dataset/videos"

TRAIN = 'videos/train'
VAL = 'videos/val'
TEST = 'videos/test'
ANNOTIONS = 'annotations'

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import seaborn as sn
import pandas as pd
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pylab import *
import time
import os
import copy
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import array as arr
from google.colab.patches import cv2_imshow
import cv2
from PIL import Image
import math
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CustomDatasetImage(Dataset):
  
  def __init__(self, root_dir, phase,transform):
    self.dataset = self.extract_pairs(root_dir + phase)
    self.root_dir = root_dir + phase
    self.phase = phase
    self.annototions = self.readAnns(root_dir)
    self.to_tensor = transforms.ToTensor()
    self.transform = transform
  
  def extract_pairs(self, dir_path):
    # [(1,2),(2,3)]
    # dir pathdeki tüm fileların pathlerini pair halinde oluşturup bir arraya at bu arrayi de döndür.
    #print(dir_path)
    pairs = []
    paths_mem = {}
    
    for file in os.listdir(dir_path):
      paths = []
      for entry in os.listdir(dir_path + '/' + file):
        entry_parsed = entry.split('.')
        #print(entry_parsed[0])
        paths.append(str(dir_path + '/' + file + '/' + entry))
      paths.sort()
      paths_mem.update({str(file):paths})
    
    for key1, value1 in paths_mem.items():
      #print(key1, value1)
      for i in range(size(value1)-1):
        pairs.append(((key1,value1[i]), (key1,value1[i+1])))
    return pairs

  def __len__(self):
    return len(self.dataset) 
    
  def __getitem__(self, index):
    pair = self.dataset[index]
    inner_key_lst = pair[0][1].split('/')
    inner_key = inner_key_lst[-1].split('.')[0].lstrip('0')
    #print(pair[0][1])
    img1 = Image.open(pair[0][1])
    width, height = img1.size
    img2 = Image.open(pair[1][1])
    tmp = self.annototions.get(pair[0][0]).get(inner_key)
    #print(tmp)
    for i in range(size(tmp)):
      tmp[i] = tmp[i].split('.')[0]
    tmp = np.array(tmp).astype(int)
    #recalculate coordinates according to resizing
    tmp[0] = 224*tmp[0]/width
    tmp[2] = 224*tmp[2]/width
    tmp[1] = 224*tmp[1]/height
    tmp[3] = 224*tmp[3]/height
    center = ((tmp[0]+tmp[2])/2,(tmp[1]+tmp[3])/2)
    #coordinates of 2 times enlarged box
    x1 = 2*tmp[0] - center[0]
    y1 = 2*tmp[1] - center[1]
    x2 = 2*tmp[2] - center[0]
    y2 = 2*tmp[3] - center[1]
    tmp2 = np.array((x1, y1, x2, y2))
    #print(tmp2)
    #crop images
    crop_img2 = img2.crop(tmp2)
    crop_img = img1.crop(tmp)
    crop_img = self.transform(crop_img)
    crop_img2 = self.transform(crop_img2)
    data = [crop_img,crop_img2, tmp2]
    
    return data
    
    
  def readAnns(self,root_dir):
    ann_dict = {}
    for entry in os.listdir(root_dir + ANNOTIONS):
        entry_parsed = entry.split('.')
        inner_dict = {}
        #print(entry_parsed[0],root_dir + ANNOTIONS + '/' + entry)
        with open(root_dir + ANNOTIONS + '/' + entry, 'r') as f:
          for line in f:
            line_parsed = line.split(' ')
            #print(line_parsed)
            inner_dict.update({line_parsed[0]:line_parsed[1:]})
        ann_dict.update({entry_parsed[0]:inner_dict})
    return ann_dict

train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
])
data_dir = "/content/drive/My Drive/Colab Notebooks/dataset/"
data = CustomDatasetImage(data_dir, TRAIN, train_transform)
train_loader = torch.utils.data.DataLoader(data, batch_size = 1,shuffle = True)
#data.__getitem__(2)

def feature_extractor(model, train_loader):
  model.to(device)
  model.eval()
  features = []
  with torch.no_grad():
    for (img1, img2, ground_truth) in train_loader:
      feature = []
      img1 = img1.to(device)
      img2 = img2.to(device)
      output = model(img1)
      output2 = model(img2)
      #print(type(output),type(ground_truth))
      pooling = nn.AvgPool2d(kernel_size = 7)
      output = pooling(output)
      output2 = pooling(output2)
      third = torch.cat((output, output2),1)
      pack = [third, ground_truth]
      features.append(pack)
  return features

vgg16 = models.vgg16(pretrained = True)
#print(vgg16)
new_vgg16 = nn.Sequential(vgg16.features)
#print(new_vgg16)

train_features = feature_extractor(new_vgg16,train_loader)
#print(size(train_features))

print(size(train_features))

class  Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.relu = nn.ReLU()
    self.fc1 = nn.Linear(1024,1024)
    self.fc2 = nn.Linear(1024,1024)
    self.fc3 = nn.Linear(1024,4)
  
  def forward(self,x):
    x = x.view(-1)
    out = self.fc1(x)
    out = self.relu(out)
    out = self.fc2(out)
    out = self.relu(out)
    out = self.fc3(out)
    return out

network = Net()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(network.parameters(), lr= 0.0001)
network.cuda()

def train_model(model, criterion, optimizer,phase, train_features,number_of_epochs):
  since = time.time()
  val_acc_history = []
  validation_loss = []
  train_loss = []
  epoch_number = []
  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 10000.0
  
  for epoch in range(number_of_epochs):
    print('Epoch {}/{}'.format(epoch, number_of_epochs - 1))
    print('-' * 10)
    epoch_number.append(epoch)
    if phase == 'train':
      model.train()
    else:
      model.eval()
    running_loss = 0.0
    
    for [feature, ground_truth] in train_features:
      #print(type(feature), type(ground_truth),shape(feature))
      ground_truth = ground_truth.float()
      feature = feature.to(device)
      ground_truth = ground_truth.to(device)
      # zero the parameter gradients
      optimizer.zero_grad()
      outputs = network(feature)
      loss = criterion(outputs, ground_truth)
      #print(loss)
      if phase == 'train':
        loss.backward()
        optimizer.step()
      running_loss += loss.item()
    epoch_loss = running_loss / len(train_features)
    
    print('{} Loss: {:.4f}'.format(phase, epoch_loss))
    print()        
    if(phase == 'train'):
          train_loss.append(epoch_loss)
    else:
          validation_loss.append(epoch_loss)
    # deep copy the model
    if phase == 'validation' and epoch_loss < best_loss:
        best_loss = epoch_loss
        best_model_wts = copy.deepcopy(model.state_dict())
    if phase == 'validation':
        train_loss.append(epoch_loss)
  
  plt.figure(figsize=(8, 6))
  plot1 = plt.plot(epoch_number, train_loss)
  plt.legend(plot1, "Loss")
  plt.xlabel('Epoch')
  plt.ylabel('Average Negative Log Likelihood')
  plt.title('Losses')

  time_elapsed = time.time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
  print('Best val loss: {:4f}'.format(best_loss))
  # load best model weights
  model.load_state_dict(best_model_wts)
  return model, train_loss

trained_model, acc = train_model(network, criterion, optimizer, 'train', train_features, 50)

data2 = CustomDatasetImage(data_dir, VAL, train_transform)
validation_loader = torch.utils.data.DataLoader(data2, batch_size = 1,shuffle = True)

val_features = feature_extractor(new_vgg16,validation_loader)

class TestDataset(Dataset):
  
  def __init__(self, root_dir,phase, transform):
    self.dataset = self.extract_pairs(root_dir + phase)
    self.root_dir = root_dir + phase
    self.phase = phase
    self.annototions = self.readAnns(root_dir)
    self.to_tensor = transforms.ToTensor()
    self.transform = transform
  
  def __getitem__(self,index):
    file_name, frames = self.dataset[index]
    data = []
    a = 0
    for i in frames:
      inner_key_lst = i.split('/')
      inner_key = inner_key_lst[-1].split('.')[0].lstrip('0')
      img1 = Image.open(i)
      width, height = img1.size
      tmp = self.annototions.get(file_name).get(inner_key)
      #print(tmp)
      for i in range(size(tmp)):
        tmp[i] = tmp[i].split('.')[0]
      tmp = np.array(tmp).astype(int)
      #recalculate coordinates according to resizing
      tmp[0] = 224*tmp[0]/width
      tmp[2] = 224*tmp[2]/width
      tmp[1] = 224*tmp[1]/height
      tmp[3] = 224*tmp[3]/height
      if a < 1:
        img1 = img1.crop(tmp)
      data.append((self.transform(img1), tmp))
        
    return data
  
  def __len__(self):
    return len(self.dataset)
  
  def readAnns(self,root_dir):
    ann_dict = {}
    for entry in os.listdir(root_dir + ANNOTIONS):
        entry_parsed = entry.split('.')
        inner_dict = {}
        #print(entry_parsed[0],root_dir + ANNOTIONS + '/' + entry)
        with open(root_dir + ANNOTIONS + '/' + entry, 'r') as f:
          for line in f:
            line_parsed = line.split(' ')
            #print(line_parsed)
            inner_dict.update({line_parsed[0]:line_parsed[1:]})
        ann_dict.update({entry_parsed[0]:inner_dict})
    return ann_dict
  
  def extract_pairs(self, dir_path):
    # [(1,2),(2,3)]
    # dir pathdeki tüm fileların pathlerini pair halinde oluşturup bir arraya at bu arrayi de döndür.
    #print(dir_path)
    paths_mem = []
    for file in os.listdir(dir_path):
      paths = []
      for entry in os.listdir(dir_path + '/' + file):
        entry_parsed = entry.split('.')
        #print(entry_parsed[0])
        paths.append(str(dir_path + '/' + file + '/' + entry))
      paths.sort()
      #print(file)
      paths_mem.append((str(file),paths))
      #print(paths[0],paths[1], paths[2])
    
    return paths_mem
  
  
  def readAnns(self,root_dir):
    ann_dict = {}
    for entry in os.listdir(root_dir + ANNOTIONS):
        entry_parsed = entry.split('.')
        inner_dict = {}
        #print(entry_parsed[0],root_dir + ANNOTIONS + '/' + entry)
        with open(root_dir + ANNOTIONS + '/' + entry, 'r') as f:
          for line in f:
            line_parsed = line.split(' ')
            #print(line_parsed)
            inner_dict.update({line_parsed[0]:line_parsed[1:]})
        ann_dict.update({entry_parsed[0]:inner_dict})
    return ann_dict

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
])
data_dir = "/content/drive/My Drive/Colab Notebooks/dataset/"
data = TestDataset(data_dir, TEST, test_transform)
test_loader = torch.utils.data.DataLoader(data, batch_size = 1,shuffle = False)

def test_feature_extractor(model, data):
  model.to(device)
  model.eval()
  features = []
  with torch.no_grad():
    for frame, ann in data:
      img = frame.to(device)
      output = model(img)
      pooling = nn.AvgPool2d(kernel_size = 7)
      output = pooling(output)
      features.append((output,ann))
  
    
  return features

def eval_model(vgg, criterion):
  since = time.time()
  avg_loss = 0
  loss_test = 0
  total = 0
  vgg.to(device)
  for data in test_loader:
    vgg.train(False)
    vgg.eval()
    features= test_feature_extractor(new_vgg16, data)
    #print(len(features))
    for i in range(len(features)-1):
      feature1 = features[i][0]
      feature2 = features[i + 1][0]
      feature1.to(device)
      feature2.to(device)
      third = torch.cat((feature1,feature2),1)
      outputs = trained_model(third)
      loss =  criterion(outputs, features[i+1][1].float().to(device))
      loss_test += loss
      total += 1
      #print(total, loss_test)
  
  avg_loss = loss_test /total
  elapsed_time = time.time() - since
  print(0)
  print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
  print("Avg loss (test): {:.4f}".format(avg_loss))

eval_model(trained_model,criterion)