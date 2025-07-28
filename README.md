<img width="966" height="114" alt="image" src="https://github.com/user-attachments/assets/c52978dd-df3a-423c-91bb-59bac65019cb" />


Official implementation of NRF-RTI: A Neural Reflectance Field Model for Accurate Relighting in RTI Applications

<img width="1420" height="386" alt="Screenshot 2025-07-28 at 20 36 52" src="https://github.com/user-attachments/assets/934486f8-87b6-45c5-a5ef-08223dbdb128" />
Our method takes as input a set of images of an object captured from various lighting directions (in this example, we used 80
input images) and produces images under novel light directions (a). Panel (b) shows the ground truth image for the selected novel
light, not present in the training set, while (c) is the error map depicting the Euclidean distance between the ground-truth and the reconstructed
images in RGB space.


# Table of Contents
- [License](#license)
- [Get Started](#get-started)
  - [Installation](#installation)
  - [Checkpoints](#checkpoints)
- [Usage](#usage)
- [Training](#training)
  - [Datasets](#datasets)
  - [Demo](#demo)  - [Our Hyperparameters](#our-hyperparameters)
- [Computing relighing accuracy](#accuracy)
- [Relighting](#relighting)
  
## License

Details...

## Get Started

Instructions...

### Installation

1. clone NRF-RTI
<pre> pip install -r requirements.txt 
python train.py --config config.yaml </pre>

2. Create an environment, for example, using conda
<pre> conda environment creation 
  ... </pre>
  
### Checkpoints
You can obtain our pre-trained models for each object by downloading them directly.

You can check the hyperparameters we used in section ### Our Hyperparameters

## Usage
<pre> Write usage code here for relighting
</pre>

## Training
In this section, we present the training dataset, preprocessing, and training demo.

## Our Hyperparameters

