import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch.autograd import Variable
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Model(nn.Module):
    def __init__(self,input_height,input_width,input_channels):
        super(Model,self).__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.layer1 = nn.Sequential(nn.BatchNorm2d(input_channels),nn.Conv2d(input_channels,32,5,stride=2,padding=2,bias=True),nn.ReLU())
        self.layer2 = nn.Sequential(nn.BatchNorm2d(32),nn.Conv2d(32,64,5,stride=1,padding=2,bias=True),nn.ReLU())
        self.layer3 = nn.Sequential(nn.BatchNorm2d(64),nn.Conv2d(64,128,5,padding=2,bias=True),nn.ReLU())
        self.layer4 = nn.Sequential(nn.BatchNorm2d(128),nn.Conv2d(128,128,5,padding=2,bias=True),nn.ReLU())
        #self.layer4 = nn.MaxPool2d(3,stride=1,padding=1)
        self.layer5  = nn.Sequential(nn.BatchNorm2d(128),nn.Conv2d(128,128,5,padding=2,bias=True),nn.ReLU())
        self.layer6 = nn.Sequential(nn.BatchNorm2d(128),nn.Conv2d(128,1,1,bias=True))
        #self.layer7 = nn.Sequential(nn.BatchNorm2d(1),nn.Conv2d(1,1,3,stride=2,padding=1,bias=True),nn.ReLU())
        #self.fc1 = nn.Sequential(nn.Linear(3072,3072,bias=True))
        
    def forward(self,input_img):
        
        output = self.layer1(input_img)
 
        output = self.layer2(output)

        output = self.layer3(output)

        output = self.layer4(output)

        output = self.layer5(output)
        
        edges = self.layer6(output)
        
        edges = F.sigmoid(edges)
        
        edges =  edges.view(edges.shape[0],edges.shape[2],edges.shape[3])
        
        return edges

