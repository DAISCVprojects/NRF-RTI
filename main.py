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
from dataloader import RTI_data as dataset
from model import RTINetwork
import random
 
arg = get_arguments()
dataset_path = None

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(52)
    torch.cuda.manual_seed(seed=52)
    
def data_path():
    return dataset_path    

def load_dataset():
    print("\n Loading the dataset \n")
    
    print("Dataset directory:", arg.dataset_dir)
    print("Save Directory:", arg.dataset_dir + 'saved_model')
    g = torch.Generator()
    transform = transforms.ToTensor()
    if arg.mode == 'train':
       # intializing the dataset
        train_set = dataset(arg.dataset_dir, torch.from_numpy, arg.mode)
        val_set = dataset(arg.dataset_dir, torch.from_numpy, mode='val')
    
        # constructing the dataloader
        train_loader = dataloader(train_set, 
                                  batch_size = arg.batch_size,
                                  num_workers=arg.workers,
                                  worker_init_fn=seed_worker,
                                  generator = g,
                                  shuffle=True)
        val_loader = dataloader(val_set, 
                                batch_size = arg.batch_size,
                                num_workers=arg.workers,
                                worker_init_fn=seed_worker,
                                generator = g,
                                shuffle=False
                                )
        return train_loader, val_loader
    
    else:
        # intializing the dataset
        test_set = dataset(arg.dataset_dir, torch.from_numpy, mode='test')
    
        # constructing the dataloader
        test_loader = dataloader(test_set,
                                 batch_size = arg.batch_size, 
                                 num_workers=arg.workers,
                                worker_init_fn=seed_worker,
                                generator = g,
                                 shuffle=False)
        return test_loader
    
    
    


def train(model, train_loader, val_loader):
    print("\n Training....\n")
    
    train_loss = []
    val_loss  = []
    
    # initializing the loss and optimizer
    #criterion = nn.MSELoss(reduction='mean')
    criterion = nn.L1Loss(reduction='mean')
    optimizer = optim.Adam(model.parameters(),
                           lr = arg.learning_rate
                           )
    
    start_epoch = 0
    best_loss = 1000
    
    # start training
    print()
    train = Train(model, train_loader, optimizer, criterion, arg.device)
    val = Test(model, val_loader, criterion, arg.device)   
    
    for epoch in range(start_epoch, arg.epochs):
        print(">>>>>>>>Epoch {0} Training".format(epoch))
        
        loss_train = train.run_epoch(epoch)
        train_loss.append(loss_train)
        
        print(">>>>Epoch {0:d} Avg.loss {1:.4f}".format(epoch, loss_train))
       
        print(">>>>>>Epoch {0} Validation".format(epoch))
        loss_val = val.run_epoch(epoch)
        val_loss.append(loss_val)
        
        print(">>>>[Epoch {0:d}] Avg.loss {1:.4f}".format(epoch, loss_val))
       
        # save the model if better than best_loss
        if loss_val<best_loss:
            best_loss = loss_val
            print()
            print(">>>>>>>>>Saving best model thus far...>>>>")
            utils.save_checkpoint(model, optimizer, epoch+1, best_loss, arg.dataset_dir + 'saved_model', arg.model_name)
            
            
    # plotting the train/val loss
    fig = plt.figure(figsize = (4,4))
    axes = fig.add_subplot(111)
    epoc = np.arange(0, arg.epochs, 1)
    axes.plot(epoc, train_loss, color = 'b', marker = 'o', label = 'train')
    axes.plot(epoc, val_loss, color = 'g', marker = '*', label = 'val')
    axes.set_title("Training")
    axes.set_xlabel('Epochs')
    axes.set_ylabel('loss')
    axes.legend(loc='upper right')
    
    fig.savefig(arg.dataset_dir + 'loss.png')
    plt.close(fig)
    
def test(model, test_loader):
    print(">>>>>>>Testing\n")  
    print()
    criterion = nn.MSELoss()  
    test = Test(model, test_loader, criterion, arg.device)  
    loss, psnr, ssim, percp, mse = test.run_epoch(None)
     
    file_metric = open(arg.dataset_dir + 'metric.txt',"w+")
    file_string = arg.dataset_dir.split('/')
    file_metric.write('>>>>>>>>>>> Metric Error for ' + file_string[-2] + '>>>>>>>>>>>>')
    file_metric.write('\n')
    file_metric.write(">>>>>>>>>Avg.loss  {0:.3f}".format(loss))
    file_metric.write('\n')
    file_metric.write(">>>>>>>>>PSNR {0:.3f}".format(np.array(psnr).mean()))
    file_metric.write('\n')
    file_metric.write(">>>>>>>>>SSIM  {0:.3f}".format(np.array(ssim).mean()))
    file_metric.write('\n')
    file_metric.write(">>>>>>>>>Perceptual_Loss  {0:.3f}".format(np.array(percp).mean()))
    file_metric.write('\n')
    file_metric.write(">>>>>>>>>MSE  {0:.3f}".format(np.array(mse).mean()))
    file_metric.close()
   
   
        
'''>>>>>>>>>>>>>>>> main starts here>>>>>>>>>>>>>>>>>>>>>>>>> '''
torch.manual_seed(52)
torch.cuda.manual_seed(seed=52)
np.random.seed(seed =52)

# checking if dataset directory exists

assert os.path.isdir(arg.dataset_dir), "The directory {0} does not exit ".format(arg.dataset_dir)
assert os.path.isdir(arg.dataset_dir + 'saved_model'), "The directory {0} does not exist".format(arg.dataset_dir + 'saved_model')
assert arg.mode.lower() in {'train', 'test'}, "Execution mode should be {0}".format(arg.mode.lower())

dataset_dir = arg.dataset_dir + 'train'
images_path =  os.listdir(dataset_dir )
# number of training images
input_len = len(images_path)
model  = RTINetwork(input_len).to(arg.device)


if arg.mode.lower() == 'train':
    train_loader, val_loader= load_dataset()
    train(model, train_loader, val_loader)
    
if arg.mode.lower() == 'test':
    test_loader  = load_dataset()
    # initialize the optimizer
    optimizer = optim.Adam(model.parameters())
    
    
    model, optimizer, epoch, loss = utils.load_checkpoint(model, optimizer, arg.dataset_dir + 'saved_model', arg.model_name)
    test(model, test_loader)
    