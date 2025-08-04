import torch
import torch.nn as nn
import numpy as np
from customeFunctions import sineActivation
import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)

class RTINetwork(nn.Module):
    def __init__(self, input_len = 50) :
        super(RTINetwork, self).__init__()
        # length of the input, if PCA used, set it to the size of PCA component otherwise length_of_dataset*3
        self.input_len = 20
        # Convolutional layers
        self.Conv_layer1 = nn.Conv2d(self.input_len, 128, kernel_size=5, stride=1, padding=5//2)  # best so far 32 channel
        self.Conv_layer2 = nn.Conv2d(128, 128,  kernel_size=5, stride=1, padding=5//2)
        self.Conv_layer3 = nn.Conv2d(128,64, kernel_size=5, stride=1, padding=5//2)
        self.Conv_layer4 = nn.Conv2d(64, 32,  kernel_size=3, stride=1, padding=3//2)
        self.Conv_layer5 = nn.Conv2d(32, 1,  kernel_size=1, stride=1)   

        # Fully connected layers , 11*11 are the patch size, change this values according to the patch size
        self.Fc_layer1 = nn.Linear(11*11*1,256)
        self.Fc_layer2 = nn.Linear(256,256)
        self.Fc_layer3 = nn.Linear(256, 256)
        self.Fc_layer4 = nn.Linear(256, 256)
        self.Fc_layer5 = nn.Linear(256, 10)
        
        # Mlp layers
        self.Mlp_layer1 = nn.Linear(39,256)
        self.Mlp_layer2 = nn.Linear(256, 256)
        self.Mlp_layer3 = nn.Linear(256, 256)
        self.Mlp_layer4 = nn.Linear(256, 256)
        self.Mlp_layer5 = nn.Linear(256, 3)
        
        
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.sin = sineActivation()
        self.relu = nn.ReLU()
    
        
    def forward(self,x1, x2):
        
        x1  = self.elu(self.Conv_layer1(x1 ))
        x1  = self.elu(self.Conv_layer2(x1))
        x1  = self.elu(self.Conv_layer3(x1))
        x1  = self.elu(self.Conv_layer4(x1))
        x1  = self.elu(self.Conv_layer5(x1))  
        x1  = torch.flatten(x1).reshape(x1.shape[0], x1.shape[1]*x1.shape[2]*x1.shape[3])
      
        x1 = self.elu(self.Fc_layer1(x1))
        x1 = self.elu(self.Fc_layer2(x1))
        x1 = self.elu(self.Fc_layer3(x1))
        x1 = self.elu(self.Fc_layer4(x1))
        x1 = self.elu(self.Fc_layer5(x1)) 
  

     
        x = torch.cat((x1, x2), dim = 1)
        x = self.elu(self.Mlp_layer1(x))
        x = self.elu(self.Mlp_layer2(x))
        x = self.elu(self.Mlp_layer3(x))
        x = self.elu(self.Mlp_layer4(x))
        x = self.sigmoid(self.Mlp_layer5(x))
       
        return x,x1
    

# if you want to use the pretrained models for the synthetic multi-material objects , replace the following two lines of code in the coresponding line in the above class. 
# there is no performance difference, but the model for these objects were trained with latent vector length of 8 instead og 10

        # In the fully connected layer 
#self.Fc_layer5 = nn.Linear(256, 10)
        
        # In the Mlp layers
#self.Mlp_layer1 = nn.Linear(39,256)