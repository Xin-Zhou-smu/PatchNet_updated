# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 14:25:28 2020

@author: xinzhou.2020
"""

## Function: Read .out file, change its format and store in .pkl file
## Input:  path of "*.out" (path_input_data)
## Output: path of "*.pkl" (path_output_data)


from preprocessing import extract_commit
import pickle

if __name__ == "__main__": 
    
    path_input_data = "test_data.out"
    path_output_data = "./try_data/test_try.pkl"
    show_data = True                    # print the generate data or not
    
    #read commits and change format into dict
    commits_ = extract_commit(path_file=path_input_data)
    nfile, nhunk, nloc, nleng = 2, 5, 10, 120
    #print(commits_[0])
    
    # store dict data
    with open(path_output_data, 'wb') as handle:
        pickle.dump(commits_, handle, protocol=pickle.HIGHEST_PROTOCOL)
   
    # print out the generated data
    if (show_data == True):
        f = open(path_output_data,'rb')
        data = pickle.load(f)
        print(data[0:2])
