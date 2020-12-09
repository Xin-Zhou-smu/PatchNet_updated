# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 14:43:29 2020

@author: xinzhou.2020
"""
# Function: read training_data.out and test_data.out, and generate dictionaries for message and code
# Input: path of test data (path_input_data1), path of training data(path_input_data1)
# Output: store dictionaries in "path_output_data"



from train import train_model
from evaluation import evaluation_model
from preprocessing import reformat_commit_code, extract_commit
import pickle 
from extracting import extract_msg, extract_code, dictionary


if __name__ == '__main__':
 
    path_input_data1 = "test_data.out"
    path_input_data2 = "training_data.out"
    path_output_data = "./try_data/dict_try.pkl"
    show_dict = True
    
    test_data = extract_commit(path_file=path_input_data1)
    train_data = extract_commit(path_file=path_input_data2)
    
    whole_data = train_data + test_data   # add  train data and test data together
    #whole_data = test_data
    msgs, codes = extract_msg(whole_data), extract_code(whole_data)
    dict_msg, dict_code = dictionary(data=msgs), dictionary(data=codes)
    
    #print(len(msgs))
    #print (msgs[1])
    #print(len(codes))
    #print(codes[1])
    
    print(len(dict_msg))
    print(len(dict_code))

    dict_whole = (dict_msg, dict_code)
    with open(path_output_data, 'wb') as handle:
        pickle.dump(dict_whole, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    if show_dict == True:
        f = open(path_output_data,'rb')
        data = pickle.load(f)
        #print(type(data)) # length is 2:  [dict_msg, dict_code]
        #print(len(data))
        #print(len(data[0]))
        #print(len(data[1]))
        #print(data[0])
        #print(data[1])
      
