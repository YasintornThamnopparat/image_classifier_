import numpy as np
import torch
from torch import nn,optim
from torchvision import datasets,transforms
import torchvision.models as models
import argparse

parser = argparse.ArgumentParser(description='Train.py')
parser.add_argument('data_directory', type=str, help='Directory containing data')
parser.add_argument('--learning_rate',dest = 'learning_rate',action='store',default = 0.002)
parser.add_argument('--epochs',dest='epochs',action='store',default = 15)
parser.add_argument('--arch',dest='arch',action='store',default='vgg16')
parser.add_argument('--hidden_layer',dest='hidden_layer',action='store',default=4096)
parser.add_argument('--step',dest='step',action='store',default=0)
parser.add_argument('--print_every',dest='print_every',action='store',default=5)
parser.add_argument('--data_dir',dest='data_dir',action='store',default = './flowers')
parser.add_argument('--dropout',dest='dropout',action='store',default= 0.3)
parser.add_argument('--gpu',dest='gpu',action='store',default='gpu')

result = parser.parse_args()
learn_rate = result.learning_rate
epochs = result.epochs
arch = result.arch
hiddenlayer = result.hidden_layer
data_dir = result.data_dir
dropout = result.dropout
gpu = result.gpu

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
    
if arch == 'vgg16':
    model = models.vgg16(pretrained=True)
elif arch == 'vgg13':
    model = models.vgg13(pretrained=True)
if gpu :
    if torch.cuda.is_available() :
        device = torch.device('cuda')
    else :
        device = torch.device('cpu')
    
from collections import OrderedDict
model.classifier = nn.Sequential(OrderedDict([
                            ('fc1',nn.Linear(25088,hiddenlayer)),
                            ('relu1',nn.ReLU()),
                            ('dropout1',nn.Dropout(p=dropout)),
                            ('fc2',nn.Linear(hiddenlayer,hiddenlayer)),
                            ('relu2',nn.ReLU()),
                            ('dropout2',nn.Dropout(p=dropout)),
                            ('fc3',nn.Linear(hiddenlayer,1000)),
                            ('relu3',nn.ReLU()),
                            ('droupout3',nn.Dropout(p=dropout)),
                            ('fc4',nn.Linear(1000,102)),
                            ('output',nn.LogSoftmax(dim=1))]))
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(),lr = learn_rate)
model.to(device)

running_loss = 0
steps = result.step
print_every = result.print_every
    
for epoch in range(epochs) :
    for images,labels in trainloader :
        steps +=1
        images,labels = images.to(device),labels.to(device)
        
        optimizer.zero_grad()
        log_ps = model.forward(images)
        loss = criterion(log_ps,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if steps % print_every == 0 :
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad() :
                for images,labels in validloader :
                    images,labels = images.to(device),labels.to(device)
                    logps = model.forward(images)
                    batch_loss = criterion(logps,labels)
                    valid_loss += batch_loss.item()
                    
                    ps = torch.exp(logps)
                    top_p,top_class = ps.topk(1,dim=1)
                    equal = top_class==labels.view(*top_class.shape)
                    accuracy += torch.mean(equal.type(torch.FloatTensor)).item()
                    
            print(f'Epoch : {epoch+1}/{epochs}..'
                 f'Train loss : {running_loss/print_every:.3f}..'
                 f'Validation loss : {valid_loss/len(validloader):.3f}..'
                 f'Test accuracy : {accuracy/len(validloader):.3f}')
            running_loss = 0
            model.train()
    
checkpoint = {
              'state_dict' : model.state_dict(),
              'optimizer' : optimizer.state_dict(),
              'images_train' : train_data.class_to_idx,
              'classifier' : model.classifier,
              'epoch' : epochs}
    
torch.save(checkpoint,'checkpoint.pth')
