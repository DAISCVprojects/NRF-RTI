import numpy as np
import torch
import torchvision
import os
from args import get_arguments


arg = get_arguments()


def save_checkpoint(model, optimizer, epoch, loss, save_dir, model_name):
    arg = get_arguments()
    model_name = model_name
    save_dir = save_dir
    
    assert(os.path.isdir(save_dir), "The directory {0} does not exist".format(save_dir))
    
    # saving the model
    model_path = os.path.join(save_dir, model_name)
    checkpoint = {
        'epoch':   epoch,
        'loss': loss,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict()
         
    }
    torch.save(checkpoint, model_path)
    
    # Save arguments
    summary_filename = os.path.join(save_dir, model_name + '_summary.txt')
    with open(summary_filename, 'w') as summary_file:
        sorted_args = sorted(vars(arg))
        summary_file.write("ARGUMENTS\n")
        for args in sorted_args:
            arg_str = "{0}: {1}\n".format(args, getattr(arg, args))
            summary_file.write(arg_str)

        summary_file.write("\nBEST VALIDATION\n")
        summary_file.write("Epoch: {0}\n". format(epoch))
        summary_file.write("Mean IoU: {0}\n". format(loss))


def load_checkpoint(model, optimizer, saved_dir, model_name):
    assert(os.path.isdir(saved_dir), "The directory{0} does not exist".format(saved_dir))
    
    saved_model_path = os.path.join(saved_dir, model_name)
    assert(os.path.isfile(saved_model_path), "The file {0} does not exist".format(saved_model_path))
    
    # Load the stored model parameters to the model instance
    checkpoint = torch.load(saved_model_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    return model, optimizer, epoch, loss