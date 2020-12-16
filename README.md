# PatchNet: Hierarchical Deep Learning-Based Stable Patch Identification for the Linux Kernel [[pdf](https://arxiv.org/pdf/1911.03576.pdf)]

## Contact
Questions and discussion are welcome: vdthoang.2016@smu.edu.sg

## Implementation Environment

Please install the neccessary libraries before running our tool:

- python==3.6.9
- torch==1.2.0
- tqdm==4.46.1
- nltk==3.4.5
- numpy==1.16.5
- scikit-learn==0.22.1

## Data & Pretrained models:

Please following the link below to download the data and pretrained models of our paper. 

- https://drive.google.com/drive/folders/1vO4eF4tma94tsBljLMvVXdG2K4sKOC3s?usp=sharing

After downloading, simply copy the data and model folders to PatchNet folder. 

## Generate Data .pkl from .out files:
please modify the paths(input and output) and run text2dict.py to generate train.pkl and test.pkl.

      $ python text2dict.py -text_path  [path of text data] -dict_path [path of the dictionary data want to store]  -print True
   Example:
      
      $ python text2dict.py -text_path  'train_data.out' -dict_path 'train.pkl'  
      $ python text2dict.py -text_path  'test_data.out' -dict_path 'test.pkl' 
      
please modify the paths(input and output) and run generate_dict.py to generate dict.pkl.

      $ python generate_dict.py -text_path1 [path of our data1] -text_path2 [path of our data2] -dict_path [path we want to store dict.pkl]
   Example:
    
      $ python generate_dict.py -text_path1 'training_data.out' -text_path2 'test_data.out' -dict_path 'dict.pkl'
   Notes:
   training_data.out is the "text format" patches as training dataset (used in trainig phase).
   
   test_data.out is the "text format" patches as test dataset (used in evaluation phase).
   
   The reason why we need evaluation data (test_data.out) is that if we only build a dictionary based on training dataset (training_data.out), there may be some words in test_data.out which never apprear in training_data.out. In this case, the generated dict.pkl is not the whole vacabulary. Considering it, I put both training data and test data to generate dict.pkl. As dict.pkl are consist of only token-id pairs, using test data will not affect the evaluation phase (no test info leak to model).

## Hyperparameters:
We have a number of different parameters

* --embedding_dim: Dimension of embedding vectors.
* --filter_sizes: Sizes of filters used by the convolutional neural network. 
* --num_filters: Number of filters. 
* --hidden_layers: Number of hidden layers. 
* --dropout_keep_prob: Dropout for training. 
* --l2_reg_lambda: Regularization rate. 
* --learning_rate: Learning rate. 
* --batch_size: Batch size. 
* --num_epochs: Number of epochs. 

## Running and evalutation
      
- To train the model for bug fixing patch classification, please follow this command: 

      $ python main.py -train -train_data [path of our data] -dictionary_data [path of our dictionary data]
  For example:
       
      $ python main.py -train -train_data 'train.pkl' -dictionary_data 'dict.pkl'
     
- To evaluate the model for bug fixing patch classification, please follow this command:
      
       $ python main.py -predict -pred_data [path of our data] -dictionary_data [path of our dictionary data] -load_model [path of our model]
  For example:     
  
       $ python main.py -predict -pred_data 'test.pkl' -dictionary_data 'dict.pkl' -load_model './snapshot/*.pt'

## Contact

Questions and discussion are welcome: vdthoang.2016@smu.edu.sg
