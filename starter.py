import os 
from args import get_arguments
import torch
import re
arg = get_arguments()
 # sorts the file in a list
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts 
'''
objects = ['pca1']
#objects = os.listdir(arg.dataset_dir)
#objects = ['Object1','Object2']
for obj in objects:
    material = sorted(os.listdir(os.path.join(arg.dataset_dir, obj)))

    for mat in material:
        dataset_path = os.path.join(arg.dataset_dir,obj, mat+'/')
        print(dataset_path)
        train_cmd = str('python main.py --mode train --dataset_dir '+dataset_path)
        test_cmd = str('python main.py --mode test --dataset_dir '+dataset_path)
        
        # train the model
        os.system(train_cmd)
    
        # test the model
        os.system(test_cmd)
'''
#objects = sorted(os.listdir(arg.dataset_dir), key = numericalSort)
objects = ['Object1','Object2','Object3','Object4','Object5','Object6','Object7','Object8','Object9']
for obj in objects:
    dataset_path = os.path.join(arg.dataset_dir,obj+'/')
    train_cmd = str('python main.py --mode train --dataset_dir '+dataset_path)
    test_cmd = str('python main.py --mode test --dataset_dir '+dataset_path)
        
    # train the model
    os.system(train_cmd)
    
    # test the model
    os.system(test_cmd)
    torch.cuda.empty_cache()
  