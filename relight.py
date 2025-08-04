import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader  as dataloader
import numpy as np
import matplotlib.pyplot as plt
import utils
from train import Train
from test import Test
import os
from args import get_arguments
from model import RTINetwork
import torch.utils.data as data
import random
import cv2
import pandas as pd


arg = get_arguments()
# model checkpoint and latent code path
data_path = arg.dataset_dir + '/'
# relighted images path
# test_light_direction contains 900 light directions generated on a sphere at different eight in a spiral manner 
light_path =arg.dataset_dir + 'inference_light_direction/'

class RTI_data(data.Dataset):
    def __init__(self, 
                 root_dir,
                 transform = None,
                 ):
        self.transform = transform
        self.root_dir = root_dir
        self.latent = np.load(root_dir)
    
    def __len__(self):
        num_entries = self.latent.shape[0]
        return num_entries
    
    def __getitem__(self, index):
        latent_vector = self.latent[index,:]

        
        latent_vector = self.transform(latent_vector)
        return latent_vector
    
    
class RTINetwork(nn.Module):
    def __init__(self, input_len = 50) :
        super(RTINetwork, self).__init__()
        # length of the input, if PCA used, set it to the size of PCA component otherwise length_of_dataset*3
        self.input_len = 147
        # Convolutional layers
        self.Conv_layer1 = nn.Conv2d(self.input_len, 32, kernel_size=5, stride=1, padding=5//2)  # best so far 32 channel
        self.Conv_layer2 = nn.Conv2d(32, 32,  kernel_size=5, stride=1, padding=5//2)
        self.Conv_layer3 = nn.Conv2d(32,32, kernel_size=5, stride=1, padding=5//2)
        self.Conv_layer4 = nn.Conv2d(32, 32,  kernel_size=3, stride=1, padding=3//2)
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
        self.relu = nn.ReLU()
    
        
    def forward(self,x):
        x = self.elu(self.Mlp_layer1(x))
        x = self.elu(self.Mlp_layer2(x))
        x = self.elu(self.Mlp_layer3(x))
        x = self.elu(self.Mlp_layer4(x))
        x = self.sigmoid(self.Mlp_layer5(x))
       
        return x
    
        
    def forward(self,x):
        x = self.elu(self.Mlp_layer1(x))
        x = self.elu(self.Mlp_layer2(x))
        x = self.elu(self.Mlp_layer3(x))
        x = self.elu(self.Mlp_layer4(x))
        x = self.sigmoid(self.Mlp_layer5(x))
       
        return x


def expand_light(light):
    light_list = []
    for i in range(len(light )):
        phi, theta = light[i]
        # harmonic base functions
        # order 1
        H1 = 1/np.sqrt(2*np.pi)
        # order 2
        H2 = np.sqrt(6/np.pi) * (np.cos(phi)*np.sqrt(np.cos(theta)- np.cos(theta)**2))
        H3 = np.sqrt(3/(2*np.pi)) * (-1 + 2 * np.cos(theta))
        H4 = np.sqrt(6/np.pi) * (np.sin(phi)*np.sqrt(np.cos(theta)-np.cos(theta)**2))
        # order 3
        H5 = np.sqrt(30/np.pi)*(np.cos(2*phi)*(-np.cos(theta) + np.cos(theta)**2))
        H6 = np.sqrt(30/np.pi)*(np.cos(phi)*(-1 + 2 * np.cos(theta)) * np.sqrt(np.cos(theta)-np.cos(theta)**2))
        H7 = np.sqrt(5/(2*np.pi))* (1-6*np.cos(theta) + 6 * np.cos(theta)**2)
        H8 = np.sqrt(30/np.pi)*(np.sin(phi)*(-1 + 2 * np.cos(theta)) * np.sqrt(np.cos(theta)-np.cos(theta)**2))
        H9 = np.sqrt(30/np.pi)*((-np.cos(theta) + np.cos(theta)**2)*np.sin(2*phi))

        light_list.append([H1, H2, H3, H4, H5, H6, H7, H8, H9])
    light= np.stack(light_list, axis = 0).astype(np.float32)
    
    return light

def expand_coordinate(mgrid):
    # Fourier encoding
    np.random.seed(seed=42)
    mean = (0, 0)
    cov = [[0.3, 0], [0, 0.3]]
    sample = np.random.multivariate_normal(mean, cov, size = 10).astype(np.float32)
    
    coord_temp = []
    for i in range(0, mgrid.shape[0]):
        #light_temp.append
        dot_prod = np.matmul(sample,mgrid[i,:].reshape(2,1))* 2* np.pi
        _sin = np.sin(dot_prod )
        _cos = np.cos(dot_prod)
        coord_temp.append(np.append(_sin, _cos))
    mgrid = np.array(coord_temp).astype(np.float32)
    return mgrid


# cartesian coordinate to spherical coordinate
def to_spherical(x, y, z):
    # Converts a cartesian coordinate (x, y, z) into a spherical one (radius, theta, phi).
    radius = np.linalg.norm(np.array([x, y, z]))
    # elevation/zenith
    theta = np.arctan2(z, np.sqrt(x * x + y * y))
    # azimuth
    phi = np.arctan2(y, x)
    return [phi, theta]
    
'''>>>>>>>>>>>>>>>> Relighting starts here>>>>>>>>>>>>>>>>>>>>>>>>> '''
# creating image coordinates
# h and w can be specified manually, but should match trained model image dimension
im = cv2.imread(os.path.join(data_path,'test/image01.jpg'))
h = im.shape[0]
w = im.shape[1]
dim = 2
tensor1 = torch.linspace(-1, 1, steps=h)
tensor2 = torch.linspace(-1, 1, steps=w)
mgrid = torch.stack(torch.meshgrid(tensor1, tensor2), dim = -1)
mgrid = mgrid.reshape(-1, dim)
mgrid = mgrid.numpy().astype(np.float32)
mgrid = expand_coordinate(mgrid)


# preprocessing the light directions
# changing the light(Testing light) from cartesian coordinate to spherical cooridinates
light_test = pd.read_csv(light_path, names = [0,1,2])
light_file_test = light_test.to_numpy()
n3 = len(light_file_test) 
light_dir = []
for i in range(n3):
    x, y, z = light_file_test[i]
    light_dir.append(to_spherical(x,y,z))
light_file_test  = np.stack(light_dir, axis = 0)
light_file_test =  expand_light(light_file_test)
    
# setting the device
device = 'cuda:0' if torch.cuda.is_available()  else 'cpu'

# loading the model
saved_model_path = os.path.join(data_path,'saved_model')
model_name = 'rtiNet'
checkpoint_path = os.path.join(saved_model_path, model_name)
model  = RTINetwork().to(device)
checkpoint = torch.load(checkpoint_path,map_location='cuda:0')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# loading encode reflectance data (encoded MLICs)
batch_size =4096
workers = 8
encoded_rf = os.path.join(data_path,'latent.npy')
encoded_latent = np.load(encoded_rf)
latent_set = RTI_data(encoded_rf, torch.from_numpy)

# constructing the dataloader
latent_loader = dataloader(latent_set,
                                 batch_size = batch_size, 
                                 shuffle=False)

print('<<<<<<<<<Relighting In progress<<<<<<<<<<<<<<<')
for lt in range(0,light_file_test.shape[0]):
    lit_ = np.array([light_file_test[lt,:]])
    # repeating the light vector batch_size times
    
    
    for step, batch in enumerate(latent_loader):
        latent_batch = batch.to(device)
        pxl_coord = torch.from_numpy(mgrid[step*latent_batch.shape[0]:(step*latent_batch.shape[0])+latent_batch.shape[0], :]).to(device)
        lt_vetcor = np.repeat(lit_, repeats = latent_batch.cpu().numpy().shape[0], axis =0).astype(np.float32)
            
        lt_vetcor = torch.from_numpy(lt_vetcor).to(device)
        input = torch.cat((latent_batch,pxl_coord,lt_vetcor), axis=1)
        output = model(input)
        if step ==0:
            output_values = output.detach().cpu().numpy()
        else:
            output_values = np.concatenate((output_values, output.detach().cpu().numpy()))
    
    b_out = output_values[:,0].reshape(h,w)
    g_out = output_values[:,1].reshape(h,w)
    r_out = output_values[:,2].reshape(h,w)  
     
    b_o = (np.clip(b_out, 0.0,1.0)*255).astype(np.uint8)
    g_o = (np.clip(g_out, 0.0,1.0)*255).astype(np.uint8)
    r_o = (np.clip(r_out, 0.0,1.0)*255).astype(np.uint8)
    img_out = cv2.merge((b_o,g_o,r_o))
    
    img_path = os.path.join(data_path,'relighted', 'image' + str(lt) + '.png')
    cv2.imwrite(img_path, img_out)
           
print()
print('<<<<<<<<<Relighting completed<<<<<<<<<<<<<<<')





# if you want to use the pretrained models for the synthetic multi-material objects , replace the following two lines of code in the coresponding line in the above class. 
# there is no performance difference, but the model for these objects were trained with latent vector length of 8 instead og 10

        # In the fully connected layer 
#self.Fc_layer5 = nn.Linear(256, 10)
        
        # In the Mlp layers
#self.Mlp_layer1 = nn.Linear(39,256)