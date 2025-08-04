import numpy as np
import torch
import skimage 
from PIL import Image 
from skimage import img_as_ubyte
from os.path import isfile, join
import cv2
import re
import os
import imageio
from torchmetrics.functional.image.lpips import learned_perceptual_image_patch_similarity as lpips
import matplotlib.pyplot as plt
import skimage.metrics  as metrics

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
#loss_fn_alex = lpips.LPIPS(net='alex')
from args import get_arguments
from padded_input_images import pad_and_stack_image



arg = get_arguments()
# training file path
path_train = arg.dataset_dir + 'train'
# relighted images path
output_path =arg.dataset_dir + 'output/'

class Test:
    def __init__(self, model, loader,  criterion, device):
        self.model = model
        self.test_loader = loader
        self.criterion = criterion
        self.device = device
        self.patch_stack_train,  self.window, self.h, self.w = pad_and_stack_image(path_train )
         
    def normaliz(self,img):
        min = img.min()
        max = img.max()
        img = 2 * ((img - min) / (max - min)) - 1
        img = torch.from_numpy(img)
        # swap the channel
        img = img.permute(-1,0,1).to(torch.float32)
        return img.unsqueeze(0)
    
    def run_epoch(self, epoch):
        output_values = np.array(None)
        target_values = np.array(None)
        self.model.eval()
        epoch_loss = 0.0
        
        psnr_ = []
        ssim_ = []
        mse_ = []
        percp_ = []
        

        for step, batch in enumerate(self.test_loader):
            # get the input and label
            coord_light = batch[0].to(device=self.device)[:,2:]
            idx = batch[0][:,:2] 
            label = batch[1].to(device=self.device).squeeze(1)
            
            # computing patches
            #patch_stack_train = torch.from_numpy(self.patch_stack_train).to(device=self.device)
            patches = []
            for i in range(len(idx)):
                patch = self.patch_stack_train[:,int(idx[i,0]):(int(idx[i,0])+ self.window), int(idx[i,1]):(int(idx[i,1])+self.window)]
                patches.append(patch)
            patch_input = torch.from_numpy(np.array(patches)).to(device=self.device)
  

      
            with torch.no_grad():
                # feeding the input to the network
                output, latent = self.model(patch_input, coord_light)
                # computing the loss
                loss = self.criterion(output, label)
                
            # keeping track of the loss
            epoch_loss += loss.item()
                
            # concatenating the predicted and target batch pixels in to one list
            if arg.mode == 'test':
                if step ==0:
                    latent_vector = latent.detach().cpu().numpy()
                    output_values = output.detach().cpu().numpy()
                    target_values = label.detach().cpu().numpy()
                else:
                    # we need the latent vector only for one pass of the training images through the encoder
                    if latent_vector.shape[0]<(self.h*self.w)-1:
                        latent_vector = np.concatenate((latent_vector, latent.detach().cpu().numpy()), axis=0)
                    else:
                        # since we are concatenating by batch size, the size of the row may exceed self.w*self.h. 
                        latent_vector = latent_vector[:(self.h*self.w),:]
                    output_values = np.concatenate((output_values, output.detach().cpu().numpy()))
                    target_values = np.concatenate((target_values, label.detach().cpu().numpy()))
                      
        if arg.mode == 'test':
            # saving latent vector
            np.save(arg.dataset_dir + '/latent', latent_vector)
            num_images = int(output_values.shape[0]/(self.h*self.w))
            y_out = output_values[:,0].reshape(num_images, self.h,self.w)
            u_out = output_values[:,1].reshape(num_images,self.h,self.w)
            v_out = output_values[:,2].reshape(num_images, self.h,self.w)
            
            y_target = target_values[:,0].reshape(num_images, self.h,self.w)
            u_target = target_values[:,1].reshape(num_images,self.h,self.w)
            v_target = target_values[:,2].reshape(num_images, self.h,self.w)
         
            for i in range(0, num_images):
                img_path = output_path + 'image' + str(i) + '.jpeg'
                y_t = (y_target[i]*255).astype(np.uint8)
                u_t = (u_target[i]*255).astype(np.uint8)
                v_t = (v_target[i]*255).astype(np.uint8)
                
                y_o = (np.clip(y_out[i], 0.0,1.0)*255).astype(np.uint8)
                u_o = (np.clip(u_out[i], 0.0,1.0)*255).astype(np.uint8)
                v_o = (np.clip(v_out[i], 0.0,1.0)*255).astype(np.uint8)
                
                img_out = cv2.merge((y_o,u_o,v_o))
                target_out = cv2.merge((y_t,u_t,v_t))
                
                #BGR_save = cv2.cvtColor(img_out, cv2.COLOR_YUV2BGR)
                img_out_rgb = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
                img_target_rgb = cv2.cvtColor(target_out, cv2.COLOR_BGR2RGB)
                
                # computing different metrics
                d = lpips(self.normaliz(img_out_rgb ), self.normaliz(img_target_rgb), net_type='squeeze', normalize=False)
                percp_.append(d.detach().numpy()) 
                pnsr_value = psnr(cv2.cvtColor(img_out_rgb, cv2.COLOR_RGB2GRAY) , cv2.cvtColor(img_target_rgb, cv2.COLOR_RGB2GRAY), data_range=255)
                smim_value = ssim(cv2.cvtColor(img_out_rgb, cv2.COLOR_RGB2GRAY) , cv2.cvtColor(img_target_rgb, cv2.COLOR_RGB2GRAY), data_range=255)
                mse_value = mse(cv2.cvtColor(img_out_rgb, cv2.COLOR_RGB2GRAY) , cv2.cvtColor(img_target_rgb, cv2.COLOR_RGB2GRAY))
                
                psnr_.append(pnsr_value)
                ssim_.append(smim_value )
                mse_.append(mse_value)
                cv2.imwrite(img_path, img_out)
                
        if arg.mode == 'test':        
            return epoch_loss/len(self.test_loader), psnr_, ssim_,  percp_, mse_
        else:
            return epoch_loss/len(self.test_loader)
                
                
               