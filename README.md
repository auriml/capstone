# Predictor of eligibility in clinical trials on cancer
Source code to train and validate two text classifier models - based on FastText vs Deep Learning using a 1-D Convolutional Neural Network with pretrained wordembeddings - on the corpus of cancer clinical trial protocols published in clinicaltrial.gov. The models classifies short free-text sentences (describing clinical information like medical history, concomitant medication, type and features of tumor, cancer therapy etc.)  as eligible or not eligible criterion to volunteer in clinical trials.  
Both models are evaluated using cross-validation with k-folds = 5 on incremental sample sizes (1K, 10K, 100K, 1M samples) and on the largest balanced set (using undersampling) with 4.01 M classified samples available (from a total of 6 M samples) . 
## Installation
### The following dependencies need to be installed
gensim                    0.13.4.1            
h5py                      2.6.0                     
Keras                     1.2.2                     
matplotlib                2.0.0                     
num2words                 0.5.4                     
numpy                     1.12.0                    
pandas                    0.19.2                    
protobuf                  3.2.0                     
pyparsing                 2.1.10                    
python                    3.6.0                         
readline                  6.2                           
scikit-learn              0.18.1                    
scipy                     0.18.1          
sklearn                   0.0                       
tensorflow-gpu            1.0.0                    
Theano                    0.8.2                    
### Steps to Build the Models
Note: Pregenerated subsample file with 1M samples is available for download in https://www.kaggle.com/auriml/eligibilityforcancerclinicaltrials  
If the pregenerated dataset is used then proceed directly to step 2, otherwise, to build a new dataset from scratch proceed to step 0
## 0. Download clinical trial protocols:
https://clinicaltrials.gov/ct2/results?term=neoplasm&type=Intr&show_down=Y
Size of file: 1.28 GB
## 1. Preprocessing: 
### 1.1 Generate bigrams
### 1.2 Generate wordembedings 
### 1.3 Build and save on disk the dataset files
## 2. Train and evaluate the models
## 2.1 FastText classifier
### 2.1.1 Run on single sample set
### 2.1.2 Run cross-validation
## 2.2 CNN text classifier
### 2.2.1 Run on single sample set
### 2.2.2 Run cross-validation
