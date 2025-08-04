import numpy as np
import torch
import matplotlib.pyplot as plt
from args import get_arguments
from padded_input_images import pad_and_stack_image


arg = get_arguments()
# training file path
path_train = arg.dataset_dir + 'train'

class Train:
    def __init__(self, model, train_loader, optimizer, criterion, device):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.patch_stack_train, self.window,self.w, self.h = pad_and_stack_image(path_train)
     
     
    def run_epoch(self, epoch):
         
         self.model.train()
         epoch_loss = 0.0
         # We found manually coding the learning rate decay gives better result
         if epoch>=0 and epoch <35: 
            lr_rate =0.001 
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_rate
         
         if epoch>=35 and epoch <45 : 
            lr_rate =0.0001 
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_rate
                
         if epoch>=45:   
              lr_rate =0.00001 
              for param_group in self.optimizer.param_groups:
                  param_group['lr'] = lr_rate
          
         for step, batch in enumerate(self.train_loader):
             
             # get the input and label pairs
             coord_light = batch[0].to(device=self.device)[:,2:]
             idx = batch[0][:,:2].numpy()  # idx used only for keeping track of pixel coordinates after reshuffling
             label = batch[1].to(device=self.device).squeeze(1)
       
            # computing patches(e.g. 11x11) arround each pixel
             patches = []
             for i in range(len(idx)):
                 patch = self.patch_stack_train[:,int(idx[i,0]):(int(idx[i,0])+ self.window), int(idx[i,1]):(int(idx[i,1])+self.window)]
                 patches.append(patch)
             # stacking the patches    
             patch_input = torch.from_numpy(np.array(patches)).to(device=self.device)
          
             # forward propagation
             output,latent = self.model(patch_input, coord_light)
    
             # computing the loss
             loss = self.criterion(output, label)

             # backpropagate the loss and update weights
             self.optimizer.zero_grad()
             loss.backward()
             self.optimizer.step()
             
             # keeping track of the losses
             epoch_loss += loss.item()

         return epoch_loss/len(self.train_loader)
             


 