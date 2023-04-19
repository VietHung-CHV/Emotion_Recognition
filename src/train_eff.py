import glob
from itertools import chain
import os
import random
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
# Training settings
batch_size = 64 #48# 32# 32 #16 #8 #
epochs = 40
lr = 3e-5
gamma = 0.7
seed = 42
device = 'cuda'
use_cuda = torch.cuda.is_available()
print(use_cuda)

ALL_DATA_DIR = '../custom_datasets/'
TRAIN_DIR = ALL_DATA_DIR + 'train'
TEST_DIR = ALL_DATA_DIR + 'test'
train_dir,test_dir=TRAIN_DIR,TEST_DIR
IMG_SIZE = 128 
train_transforms = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ]
)

def set_parameter_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad
        
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transforms)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs) 

print(len(train_dataset), len(test_dataset))

(unique, counts) = np.unique(train_dataset.targets, return_counts=True)
(_, counts_test) = np.unique(test_dataset.targets, return_counts=True)
cw=1/counts
cw/=cw.min()
class_weights = {i:cwi for i,cwi in zip(unique,cw)}
print(counts, class_weights.values())

num_classes=len(train_dataset.classes)
print(num_classes)

weights = torch.FloatTensor(list(class_weights.values())).cuda()
if False:
    criterion = nn.CrossEntropyLoss(weight=weights)
    #criterion = nn.CrossEntropyLoss()
else:
    def label_smooth(target, n_classes: int, label_smoothing=0.1):
        # convert to one-hot
        batch_size = target.size(0)
        target = torch.unsqueeze(target, 1)
        soft_target = torch.zeros((batch_size, n_classes), device=target.device)
        soft_target.scatter_(1, target, 1)
        # label smoothing
        soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
        return soft_target

    def cross_entropy_loss_with_soft_target(pred, soft_target):
        #logsoftmax = nn.LogSoftmax(dim=-1)
        return torch.mean(torch.sum(- weights*soft_target * torch.nn.functional.log_softmax(pred, -1), 1))

    def cross_entropy_with_label_smoothing(pred, target):
        soft_target = label_smooth(target, pred.size(1)) #num_classes) #
        return cross_entropy_loss_with_soft_target(pred, soft_target)

    criterion=cross_entropy_with_label_smoothing
# criterion = nn.CrossEntropyLoss(weight=weights)

"""Train""" 
from robust_optimization import RobustOptimizer
import copy
def train(model,n_epochs=epochs, learningrate=lr, robust=False):
    # optimizer
    if robust:
        optimizer = RobustOptimizer(filter(lambda p: p.requires_grad, model.parameters()), optim.Adam, lr=learningrate)
        #print(optimizer)
    else:
        optimizer=optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learningrate)
    # optimizer=optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learningrate)
    # scheduler
    #scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    best_acc=0
    best_model=None
    for epoch in range(n_epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        model.train()
        for data, label in tqdm(train_loader):
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)

            if robust:
                #optimizer.zero_grad()
                loss.backward()
                optimizer.first_step(zero_grad=True)
  
                # second forward-backward pass
                output = model(data)
                loss = criterion(output, label)
                loss.backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            acc = (output.argmax(dim=1) == label).float().sum()
            epoch_accuracy += acc
            epoch_loss += loss
        epoch_accuracy /= len(train_dataset)
        epoch_loss /= len(train_dataset)
        
        model.eval()
        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in test_loader:
                data = data.to(device)
                label = label.to(device)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().sum()
                epoch_val_accuracy += acc
                epoch_val_loss += val_loss
        epoch_val_accuracy /= len(test_dataset)
        epoch_val_loss /= len(test_dataset)
        print(
            f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
        )
        if best_acc<epoch_val_accuracy:
            best_acc=epoch_val_accuracy
            best_model=copy.deepcopy(model.state_dict())
        #scheduler.step()
    
    if best_model is not None:
        model.load_state_dict(best_model)
        print(f"Best acc:{best_acc}")
        model.eval()
        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in test_loader:
                data = data.to(device)
                label = label.to(device)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().sum()
                epoch_val_accuracy += acc
                epoch_val_loss += val_loss
        epoch_val_accuracy /= len(test_dataset)
        epoch_val_loss /= len(test_dataset)
        print(
            f"val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
        )
    else:
        print(f"No best model Best acc:{best_acc}")
        
""""FINETUNE CNN"""
import timm
model=timm.create_model('efficientnet_b0', pretrained=False)
model.classifier=torch.nn.Identity()
model.load_state_dict(torch.load('../models/pretrained_faces/state_vggface2_enet0_new.pt'))
model.classifier=nn.Sequential(nn.Linear(in_features=1280, out_features=num_classes))
model=model.to(device)
# print(model)

set_parameter_requires_grad(model, requires_grad=False)
set_parameter_requires_grad(model.classifier, requires_grad=True)
train(model,3,0.001,robust=True)

set_parameter_requires_grad(model, requires_grad=True)
train(model,9,1e-4,robust=True)

PATH='../models/affectnet_emotions/enet_b0_5_best.pt'
# Save
torch.save(model, PATH)

# Load
print(PATH)
model = torch.load(PATH)
model=model.eval()

class_to_idx=train_dataset.class_to_idx
print(class_to_idx)
idx_to_class={idx:cls for cls,idx in class_to_idx.items()}
print(idx_to_class)

print(test_dir)
y_val,y_scores_val=[],[]
model.eval()
for class_name in tqdm(os.listdir(test_dir)):
    if class_name in class_to_idx:
        class_dir=os.path.join(test_dir,class_name)
        y=class_to_idx[class_name]
        for img_name in os.listdir(class_dir):
            filepath=os.path.join(class_dir,img_name)
            img = Image.open(filepath)
            img_tensor = test_transforms(img)
            img_tensor.unsqueeze_(0)
            scores = model(img_tensor.to(device))
            scores=scores[0].data.cpu().numpy()
            #print(scores.shape)
            y_scores_val.append(scores)
            y_val.append(y)

y_scores_val=np.array(y_scores_val)
y_val=np.array(y_val)
print(y_scores_val.shape,y_val.shape)

y_pred=np.argmax(y_scores_val,axis=1)
acc=100.0*(y_val==y_pred).sum()/len(y_val)
print(acc)

y_train=np.array(train_dataset.targets)

for i in range(y_scores_val.shape[1]):
    _val_acc=(y_pred[y_val==i]==i).sum()/(y_val==i).sum()
    print('%s %d/%d acc: %f' %(idx_to_class[i],(y_train==i).sum(),(y_val==i).sum(),100*_val_acc))
    
labels=list(class_to_idx.keys())
print(labels)

plt.subplot()
plt.bar(labels, counts, 0.5)
plt.suptitle('Custom Dataset - Training')
plt.savefig('../res/train_dts.jpg')
plt.clf()
plt.subplot()
plt.bar(labels, counts_test, 0.5)
plt.suptitle('Custom Dataset - Testing')
plt.savefig('../res/test_dts.jpg')
plt.clf()

IC = type('IdentityClassifier', (), {"predict": lambda i : i, "_estimator_type": "classifier"})
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
def plt_conf_matrix(y_true,y_pred,labels):
    print(y_pred.shape,y_true.shape, (y_pred==y_true).mean())

    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay.from_estimator(IC, y_pred,y_true,display_labels=labels,cmap=plt.cm.Blues,ax=ax,colorbar=False)
    plt.tight_layout()
    plt.savefig('../res/confusion_matrix.jpg')
plt_conf_matrix(y_val,y_pred,labels)




