# Predictor of eligibility in clinical trials on cancer
Source code to train and validate two text classifier models - based on FastText vs Deep Learning using a 1-D Convolutional Neural Network with pretrained wordembeddings - on the corpus of cancer clinical trial protocols published in clinicaltrial.gov. The models classifies short free-text sentences (describing clinical information like medical history, concomitant medication, type and features of tumor, cancer therapy etc.)  as eligible or not eligible criterion to volunteer in clinical trials.  
Both models are evaluated using cross-validation with k-folds = 5 on incremental sample sizes (1K, 10K, 100K, 1M samples) and on the largest balanced set (using undersampling) with 4.01 M classified samples available (from a total of 6 M samples) . 

The report is available in CapstoneReport-MLNanodegree.pdf
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
tensorflow-gpu            1.0.0    (note: GPU installation is optional, but highly recommended to train the CNN model in less than 5 hours)                
                    
### Steps to Build the Models
Note: A pregenerated subsample file with 1M samples and 186 MB is available for download in https://www.kaggle.com/auriml/eligibilityforcancerclinicaltrials  
If the pregenerated subsample dataset is used rename it and save it as './textData/labeledEligibility.csv' and proceed directly to step 2, otherwise, to build a new dataset from scratch proceed to step 0
## 0. Download clinical trial protocols:
https://clinicaltrials.gov/ct2/results?term=neoplasm&type=Intr&show_down=Y

Size of folder "search_result": 1.28 GB
## 1. Preprocessing: 
### 1.1 Generate bigrams
```python preprocessor.py -b '<pathTo>/search_result/'```
### 1.2 Build and save on disk the dataset file with labeled clinical statements
```python preprocessor.py -l ```
### 1.3 Train word2vects
#### 1.3.1 Using FasText
```python preprocessor.py -w ```

```python fasttext_word_embeddings.py```
##### 1.3.1 Using Gensim
```python gensim_word_embeddings.py -i '<pathTo>/search_result/'```

To visualize them using the TensorBoard (https://www.tensorflow.org/versions/master/how_tos/embedding_viz/) execute this script to produce the files in tensor format: 

```python word2vec2tensor.py --input wordEmbeddings/vectorsGensim_cbow.bin --output word2vec2tensor ```
## 2. Train and evaluate the text classifier models
## 2.1 FastText classifier
### 2.1.1 Run cross-validation using sample sizes = [1000, 10000, 100000, 1000000]
Make sure that exists the file ./textData/labeledEligibility.csv

Execute:
```python fasttext_text_classifier.py ```

The learning curves are saved in ./Learning_Curves_FastText_Classifier_plot.png
## 2.2 CNN text classifier
### 2.2.2 Run cross-validation using sample sizes = [1000, 10000, 100000, 1000000]
Make sure that exists the file ./textData/labeledEligibility.csv

Using tensorflow-gpu it takes aprox 5 hours to train.
```python text_classifier.py ```

The learning curves are saved in ./Learning_Curves_CNN_Classifier_plot.png
