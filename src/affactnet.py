import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

from PIL import Image

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image

from torchvision import datasets, transforms
from tqdm import tqdm
from model1 import Model1
# from models.custom_model import resnet50
from c_model_1 import EModel
from resnet_ver2 import *
from custom_model import Face_Emotion_CNN

print(f"Torch: {torch.__version__}")

ALL_DATA_DIR = '../datasets/01_FER2013_datasets/'
train_dir = ALL_DATA_DIR + 'train'
test_dir = ALL_DATA_DIR + 'test'
# INPUT_SIZE = (224, 224)


# Training settings
batch_size = 32 #64 #48# 32# 32 #16 #8 #
epochs = 60 #40
lr = 1e-5
gamma = 0.7
seed = 42
device = 'cuda'
use_cuda = torch.cuda.is_available()
# print(use_cuda)
# use_cuda = False


# print(train_dir,test_dir)

USE_ENET2=True #False #

IMG_SIZE= 224# 300 # 80 #
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
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ]
)


# train_transforms = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225]),
#         transforms.RandomErasing(scale=(0.02, 0.25)) ])
    
# test_transforms = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.Grayscale(3),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                             std=[0.229, 0.224, 0.225])])
# print(test_transforms)

#adapted from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
def set_parameter_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad
        
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transforms)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs) 

# print(len(train_dataset), len(test_dataset))

(unique, counts) = np.unique(train_dataset.targets, return_counts=True)
(_, counts_test) = np.unique(test_dataset.targets, return_counts=True)
cw=1/counts
cw/=cw.min()
class_weights = {i:cwi for i,cwi in zip(unique,cw)}
# print(counts, class_weights.values())

num_classes=len(train_dataset.classes)
# print(num_classes)

# loss function
weights = torch.FloatTensor(list(class_weights.values())).cuda() if use_cuda==True else torch.FloatTensor(list(class_weights.values()))

criterion = nn.CrossEntropyLoss(weight=weights)
# criterion = nn.NLLLoss()

import copy
def train(model,n_epochs=epochs, learningrate=lr, robust=False):
    
    optimizer=optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learningrate)
    # scheduler
    # scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    best_acc=0
    best_model=None
    for epoch in range(n_epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        model.train()
        for data, label in tqdm(train_loader):
            if use_cuda:
                data = data.to(device)
                label = label.to(device)

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
                if use_cuda ==True:
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
                if use_cuda:
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
        
"""FINETUNE CNN"""

import timm

# model=timm.create_model('efficientnet_b0', pretrained=False)
# model.classifier=torch.nn.Identity()
# if use_cuda==True:
#     model.load_state_dict(torch.load('../models/pretrained_faces/state_vggface2_enet0_new.pt')) #_new
#     # model.load_state_dict(torch.load('../models/pretrained_faces/state_vggface2_enet2.pt'))
# else:
#     model.load_state_dict(torch.load('../models/pretrained_faces/state_vggface2_enet0_new.pt', map_location=torch.device('cpu'))) #_new
    # model.load_state_dict(torch.load('../models/pretrained_faces/state_vggface2_enet2.pt', map_location=torch.device('cpu'))) #_new

# model.classifier=nn.Sequential(nn.Linear(in_features=1280, out_features=num_classes)) #1792 #1280 #1536
#model.head.fc=nn.Linear(in_features=3072, out_features=num_classes)
#model.head=nn.Sequential(nn.Linear(in_features=768, out_features=num_classes))
# model = ResNet50(num_classes=5)
model = timm.create_model('resnet50',num_classes=5, pretrained=True)
# model = Model1(num_features=IMG_SIZE, num_classes=5)
# model = EModel()

# res50 = ResNet(block=BasicBlock, layers=[3, 4, 6, 3], num_classes=5)
# model = res50

# model = Face_Emotion_CNN()

# model = EModel2()

if use_cuda:
    model=model.to(device)
print(model)
model.eval()
# set_parameter_requires_grad(model, requires_grad=False)
# set_parameter_requires_grad(model.classifier, requires_grad=True)
train(model,epochs,lr,robust=True)
#Best acc:0.48875007033348083
#7: Best acc:0.558712363243103
#5: Best acc:0.6665414571762085

# set_parameter_requires_grad(model, requires_grad=True)
# train(model,6,1e-4,robust=True)
#Best acc:0.8260869383811951
#val_loss : 0.0212 - val_acc: 0.8261

# set_parameter_requires_grad(model, requires_grad=False)
# set_parameter_requires_grad(model.classifier, requires_grad=True)
# train(model, 6, 0.001,robust=True)

# if USE_ENET2:
#     if False: # 7 emotions
#         PATH='../models/affectnet_emotions/enet_b2_7.pt'
#         model_name='enet2_7_pt'
#     else:
#         #PATH='../models/affectnet_emotions/enet_b2_8.pt'
#         PATH='../models/affectnet_emotions/enet_b0_5_best.pt'
#         model_name='enet0_5_pt'
# else:
#     if False: # 7 emotions from AFFECT_IMG_SEVEN_TRAIN_DATA_DIR and AFFECT_IMG_SEVEN_VAL_DATA_DIR
#         PATH='../models/affectnet_emotions/enet_b0_7.pt'
#         model_name='enet0_7_pt'
#     else:
#         PATH='../models/affectnet_emotions/enet_b0_5_best_vgaf.pt'
#         #PATH='../models/affectnet_emotions/enet_b0_8_best_afew.pt'
#         model_name='enet0_5_pt'
# print(PATH)
PATH='../models/affectnet_emotions/custom_model.pt'
# Save
torch.save(model, PATH)
print(model)

# Load
# print(PATH)
model = torch.load(PATH)
model=model.eval()

class_to_idx=train_dataset.class_to_idx
print(class_to_idx)
idx_to_class={idx:cls for cls,idx in class_to_idx.items()}
print(idx_to_class)

print(f'testd_dir: {test_dir}')
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
    # plot_confusion_matrix(IC, y_pred,y_true,display_labels=labels,cmap=plt.cm.Blues,ax=ax,colorbar=False) #,normalize='true'
    ConfusionMatrixDisplay.from_estimator(IC, y_pred,y_true,display_labels=labels,cmap=plt.cm.Blues,ax=ax,colorbar=False)
    plt.tight_layout()
    plt.savefig('../res/confusion_matrix.jpg')
    # plt.show()
plt_conf_matrix(y_val,y_pred,labels)

