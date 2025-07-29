<img width="966" height="114" alt="image" src="https://github.com/user-attachments/assets/c52978dd-df3a-423c-91bb-59bac65019cb" />


Official implementation of NRF-RTI: A Neural Reflectance Field Model for Accurate Relighting in RTI Applications

<img width="1420" height="386" alt="Screenshot 2025-07-28 at 20 36 52" src="https://github.com/user-attachments/assets/934486f8-87b6-45c5-a5ef-08223dbdb128" />
During training, our method takes as input a set of images of an object captured from various lighting directions (in this example, we used 80
input images) and produces images under novel light directions (a). Panel (b) shows the ground truth image for the selected novel
light, not present in the training set, while (c) is the error map depicting the Euclidean distance between the ground-truth and the reconstructed
images in RGB space.


# Table of Contents
- [License](#license)
- [Get Started](#get-started)
  - [Installation](#installation)
  - [Checkpoints](#checkpoints)
- [Training](#training)
  - [Datasets](#datasets)
  - [Preprocessing](#preprocessing)
  - [Our Hyperparameters](#our-hyperparameters)
- [Testig](#testing)
- [Relighting](#relighting)
  
## License

Details...

## Get Started


### Installation

1. clone NRF-RTI
<pre>  
git clone --recursive https://github.com/DAISCVprojects/NRF-RTI
cd NRF-RTI
# If you have already cloned NRF-RTI:
# git submodule update --init --recursive
 </pre>

2. Create an environment, for example, using conda
<pre> 
conda create -n RTI python=3.10.9
conda activate RTI
pip install -r requirements.txt
</pre>
  
### Checkpoints
You can obtain our pre-trained models for each object by downloading them directly.

You can check the hyperparameters we used in section [Our Hyperparameters](#our-hyperparameters)

## Training
In this section, we present the training dataset, preprocessing, and training demo.

### Datasets
We train our method on our synthetic datasets and real and synthetic datasets of [NeuralRTI]:https://github.com/Univr-RTI/NeuralRTI 

### Preprocessing
<pre> # Put the raw dataset and light direction in /NRF-RTI/datasets/
# Preprocess the datasets in HD5Y format
# You can look at the sample raw and preprocessed data in the datasets directory
python data_preprocess.py
   </pre>
   
### Our Hyperparameters
The following are the hyperparameters for training our method.
<pre>
  python main.py \
        --mode train \
        --dataset_dir "path to your dataset" \
        --batch_size 4096 \
        --learning_rate 0.01 \
        --epochs 35
  
 # For example, for training on Object1 dataset   
 python main.py --mode train --dataset_dir /datasets/object1/ --batch_size 4096 --learning_rate 0.01 --epochs 35
</pre>

## Testing
To test our method on test images and compute evaluation metrics
<pre>
  python main.py \
      --mode test \
      --dataset_dir "path to your dataset" 
      --output_dir "path to saving output dir"
  # For example, for testing on Object1 dataset
  python main.py --mode test --dataset_dir /datasets/object1/ --output_dir /datasets/object1/output/
</pre>

## Relighting
For relighting an object from any arbitrary light direction (requires only the light direction)
<pre>
  python relight.py --dataset_dir "path where model checkpoint, light direction, and latent code are saved"
  # For example, for relighting on a random light direction 
  # assuming that checkpoint, latent code, and light direction are saved in /datasets/object1/
  python relight.py --dataset_dir /datasets/object1/
</pre>
