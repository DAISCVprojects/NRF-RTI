from argparse import ArgumentParser

def get_arguments():
    "defines command line arguments and parse them"
    parser = ArgumentParser()
    
    # Execution mode
    parser.add_argument(
        "--mode",
        '-m',
        default = "train",
        choices= ["train", "test"],
        help = "train: performs training and validating the program, test performs testing the program"
        
    )

    parser.add_argument(
        "--resume",
        action = "store_true",
        help = "loads the model and resume it"
        
    )
    
    # hyperparametrs
    parser.add_argument(
        "--batch_size",
        '-b',
        type=int,
        default = 4096,  #4096 for our dataset
        help = "sets the batch size"
    )
    
    parser.add_argument(
        "--epochs",
        '-e',
        type=int,
        default = 50, #50
        help = "number of of training epochs"
    )
    # pca
    parser.add_argument(
        "--pca",
        '-p',
        default = 20, #50
        help = "number of of pca components"
    )
    parser.add_argument(
      "--learning_rate",
      '--lr',
      type=float,
      default= 0.01,  # 0.01
      help = "sets the learning rate"
      
    )
 
    # Dataset and storage
    parser.add_argument(
        "--dataset_dir",
        '-dr',
        type = str,
        default= "/NRF-RTI/Dataset/synthOur/Object1/",
        help = "raw dataset and light file directory for preprocessing the dataset in HD5Y format"
        
    )
    
    parser.add_argument(
        "--output_dir",
        '-o',
        type = str,
        default= "/NRF-RTI/Dataset/synthOur/Object1/output/",
        help = "relighted images directory"
        
    )
    
    
    parser.add_argument(
        "--model_name",
        type = str,
        default = "rtiNet",
        help = "specifies the name of the saved model"
    )
    
    # Settings
    parser.add_argument(
        "--device",
        default = "cuda",
        help = "Device for the network to be run"
        
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=8, # initial value was 0
        help="Number of subprocesses to use for data loading. Default: 4")
  
    # storage setting
    parser.add_argument(
        "--save_dir",
        type= str,
        default= "/NRF-RTI/Dataset/synthOur/Object1/saved_model"
    )
 
    return parser.parse_args()
    
    
    
    
    
    
    
    