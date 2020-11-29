import argparse
import numpy as np
import torch
from torch import optim,nn
from torchvision import models,transforms,datasets
from PIL import Image
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description = 'Predict.py')
parser.add_argument('data_directory', type=str, help='Directory containing data')
parser.add_argument('--filepath',dest='filepath',action='store',default = './checkpoint.pth')
parser.add_argument('--gpu',dest = 'gpu',action = 'store',default = 'gpu',type = str)
parser.add_argument('--arch',dest = 'arch',action = 'store',default = 'vgg16',type = str)
parser.add_argument('--imagepath',dest='imagepath',default = './flowers/test/100/image_07939.jpg',action = 'store',type = str)
parser.add_argument('--topk',dest='topk',action='store',default=5)
parser.add_argument('--learning_rate',dest = 'learning_rate',action='store',default = 0.002)
parser.add_argument('--data_dir',dest='data_dir',action='store',default = './flowers')


result = parser.parse_args()
filepath = result.filepath
gpu = result.gpu
arch = result.arch
image_path = result.imagepath
topk = result.topk
lr = result.learning_rate
data_dir = result.data_dir

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.229,0.224,0.225],
                                                          [0.485,0.456,0.406])])
valid_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.229,0.224,0.225],
                                                          [0.485,0.456,0.406])])
test_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.229,0.224,0.225],
                                                          [0.485,0.456,0.406])])
train_data = datasets.ImageFolder(train_dir,transform = train_transforms)
valid_data = datasets.ImageFolder(valid_dir,transform = valid_transforms)
test_data = datasets.ImageFolder(test_dir,transform = test_transforms)
trainloader = torch.utils.data.DataLoader(train_data,batch_size=64,shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data,batch_size=64)
testloader = torch.utils.data.DataLoader(test_data,batch_size=64)
import json
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
if gpu : 
    if torch.cuda.is_available() :
        device = torch.device('cuda')
    else :
        device = torch.device('cpu')
    
def load_checkpoint(filepath) :
    checkpoint = torch.load(filepath)
    if arch == 'vgg16' :
        model = models.vgg16(pretrained = True)
    elif arch == 'vgg13' :
        model = models.vgg13(pretrained = True)
    for param in model.parameters() :
        param.requires_grad = False
        
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.epochs = checkpoint['epoch']
    optimizer = checkpoint['optimizer']
    model.class_to_idx = train_data.class_to_idx
    return model

model = load_checkpoint(filepath)

def process_image(image):
    im =Image.open(image)
    transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    np_image = np.array(transform(im))

    return np_image

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    image = image.transpose((1, 2, 0))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, filepath, topk=5):
    model = load_checkpoint(filepath)
    model.to('cpu')
    model.eval();
    image_tensor = torch.from_numpy(process_image(image_path)).type(torch.FloatTensor).\
    unsqueeze_(dim=0).to('cpu')
    log_ps = model.forward(image_tensor)
    ps = torch.exp(log_ps)
    top_p = (ps.topk(topk,dim=1)[0].detach().numpy())[0]
    top_class = (ps.topk(topk,dim=1)[1].detach().numpy())[0]
    prob,name = [],[]
    for i in top_p :
        prob.append(i)
    for i in top_class :
        name.append(i)
    prob_name = dict()
    for i in range(len(name)) :
        prob_name[name[i]] = prob[i]
    prob_name = dict(sorted(prob_name.items(),key=lambda item:item[1],reverse = True))
    prob,name = [],[]
    for i in prob_name.keys() :
        name.append(i)
    for i in prob_name.values() :
        prob.append(i)    
    return prob,name

def show_result(image_path,prob,name):
    from PIL import Image
    im =Image.open(image_path)
    f,ax = plt.subplots(2,figsize=(8,8))
    f.suptitle(name[0])
    ax[0].imshow(im)
    ax[1].barh(name,prob)
    
prob,flower_class = predict(image_path, filepath, topk)
name = []
for i in flower_class :
    name.append(cat_to_name[i.astype(str)])
show_result(image_path,prob,name)

