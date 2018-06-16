# -*- coding: utf-8 -*-
"""
# ------------------------------------------------------------
# Auth: Abhijeet Chopra
# CWID: 50180612
# Date: Wed Nov 29, 2017
# Desc: Evaluation Script Version 3.0
# Prog: evaluation_v3.py

# REFERENCES
# ------------------------------------------------------------
# 1. http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

# REQUIRED PACKAGES
# ------------------------------------------------------------
# 1. csv
# 2. glob
# 3. os
# 4. re
# 5. numpy
# 6. sklearn

# INSTRUCTIONS
# ------------------------------------------------------------
# 1. Required 'output' folder containing .out files with predicted values
# 2. Required 'gold' folder containing .ann files with true values
# 3. Required 'eval' folder to store .csv files with evaluation results

"""

import csv
import glob
import os
import re
import numpy as np
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score

# sets the current directory
os.chdir("C:/Users/Vidhya Moorthy/Desktop/Hybrid Keyword Extraction")

# prints the current working directory
print(os.getcwd())

# prints the directory contents
os.listdir('.')

# global variables
DEBUG=False # true when debugging to display print messages

# searches value 'value' in array 'array'
def searchArray(array, value):
    # return true if element found in array
    if value in array:
        return 1
    else:
        return 0
    return -1


# prints evaluation report
def evaluate(TXT_DIR_OUT_,file_name_string):
    # evaluate file 'file_name_string' present in directory 'TXT_DIR_OUT_'
    
    # generating paths
    # ----------------------------------------      
    TXT_DIR_OUT = TXT_DIR_OUT_ # output - contains .out files with predicted values
    TXT_DIR_GOLD = "gold/"     # gold   - contains .ann files with true values
    TXT_DIR_EVAL = "eval/"     # eval   - contains .csv files with evaluation results
    
    out_file_path_ = TXT_DIR_OUT + file_name_string + '.out'
    gold_file_path_ = TXT_DIR_GOLD + file_name_string + '.ann' 
    eval_file_path_ = TXT_DIR_EVAL + 'eval' + '_' + TXT_DIR_OUT_[:-1] + '.csv'
    
    
    # readPred
    # ----------------------------------------
    y_pred_np = np.loadtxt(out_file_path_, dtype='str')
    print(y_pred_np) if(DEBUG) else None
    #print('\n--------------------readPred\n',y_pred_np)
    
    # readTrue
    # ----------------------------------------
    f = open(gold_file_path_,mode="r",encoding="utf-8").read()    
    f = re.sub('\*[^\n]+\n', '', f, flags=re.DOTALL) # removing lines starting with '*' using regular expression    
    lx = f.split("\n")
    l = lx[:-1].copy() # removing empty last line/element
    keyword_gold=[]
    
    for r in l:
        t = r.split("\t")
        print("\t" + t[2]) if(DEBUG) else None
        keyword_gold.append(t[2])
    
    keyword_gold = " ".join(keyword_gold).split() # splitting list of phrases into seperate words list 'keyword_gold'
    
    y_true_np = np.array(keyword_gold)
    print(y_true_np) if(DEBUG) else None
    
    #print('\n--------------------readTrue\n',y_true_np)
    
    # evaluation
    # ----------------------------------------
    print("\nPrinting arrays with predicted and true keywords respectively...") if(DEBUG) else None
    print(y_pred_np) if(DEBUG) else None
    print(y_true_np) if(DEBUG) else None
    
    # creating unique arrays
    y_pred_np = np.unique(y_pred_np.flat)
    y_true_np = np.unique(y_true_np.flat)
    
    print("\nCreating unique arrays...") if(DEBUG) else None
    print(y_pred_np) if(DEBUG) else None
    print(y_true_np) if(DEBUG) else None
    
    # TODO: tokenize y_true_np to single words instead of phrases
    
    # removing extra words from prediction set that were not in true set
    y_pred_np = np.intersect1d(y_pred_np, y_true_np)
    
    print("\nRemoving extra words from prediction set that were not in true set...") if(DEBUG) else None
    print(y_pred_np) if(DEBUG) else None
    print(y_true_np) if(DEBUG) else None
    
    # sorting numpy arrays
    y_pred_np.sort()
    y_true_np.sort()
    
    print("\nPrinting sorted arrays arrays...") if(DEBUG) else None
    print(y_pred_np) if(DEBUG) else None
    print(y_true_np) if(DEBUG) else None
    
    # creating a temp empty numpy array
    y_pred_temp = np.array([])
    print("\nPrinting empty prediction bucket...") if(DEBUG) else None
    print(y_pred_temp) if(DEBUG) else None
        
    # for every keyword 'x' that exists in numpy array 'y_true_np'
    for x in y_true_np:
        # if x exists in y_true, then add in y_pred_temp
        if(searchArray(y_pred_np, x)==1):
            y_pred_temp = np.append(y_pred_temp, x)
        
        # else, add dummy value in y_pred_temp
        else:
            y_pred_temp = np.append(y_pred_temp, '*')
    
    # y_pred_temp numpy array contains predicted keywords to be compared to true keywords
    print('') if(DEBUG) else None
    print('y_pred_np     : ',y_pred_np) if(DEBUG) else None
    print('y_pred_temp   : ',y_pred_temp) if(DEBUG) else None
    print('y_pred_np     : ',y_true_np) if(DEBUG) else None
    
    # calculating scores
    p_s = precision_score(y_true_np, y_pred_temp, average='micro')
    r_s = recall_score(y_true_np, y_pred_temp, average='micro')
    f_s = f1_score(y_true_np, y_pred_temp, average='micro')
    a_s = accuracy_score(y_true_np, y_pred_temp)
    
    # converting type 'numpy.float64' to type 'float'
    p_s = p_s.tolist()
    r_s = r_s.tolist()
    f_s = f_s.tolist()
    a_s = a_s.tolist()
    
    # printing report
    print('') if(DEBUG) else None
    print('Precision : {}'.format(p_s)) if(DEBUG) else None
    print('Recall    : {}'.format(r_s)) if(DEBUG) else None
    print('F-score   : {}'.format(f_s)) if(DEBUG) else None
    print('Accuracy  : {}'.format(a_s)) if(DEBUG) else None
    
    
    # updating evaluation_report.csv
    # ----------------------------------------
    header = ['filename', 'precision', 'recall', 'fscore', 'accuracy']
    row = [file_name_string, p_s, r_s, f_s, a_s]    
        
    if not os.path.exists(eval_file_path_):
        print('File \''+eval_file_path_+'\' doesn\'t exist!') if(DEBUG) else None
        print('Creating file with headers and appending data...') if(DEBUG) else None
        with open(eval_file_path_, mode='a', encoding="utf-8", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(row)
        print('Done.') if(DEBUG) else None
            
    else:
        print('File \''+eval_file_path_+'\' already exists!') if(DEBUG) else None
        print('Appending data...') if(DEBUG) else None
        with open(eval_file_path_, mode='a', encoding="utf-8", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        print('Done.') if(DEBUG) else None
    
    return None

def evaluateModelOutput(TXT_DIR_OUT_):
    # evaluate all files in directory 'TXT_DIR_OUT_'
    
    print('\nEvaluating files in \"',TXT_DIR_OUT_,'\"',':')
    print('----------------------------------------')
    
    # getting a list of files
    file_list = glob.glob(TXT_DIR_OUT_+"*.out")
    
    # iterate over the list getting each file 
    for file_name in file_list:
        file_name = file_name[:-4] # removing file extention suffix
        file_name = file_name.replace(TXT_DIR_OUT_[:-1]+'\\','') # removing dir prefix
        print(file_name,'...')
        evaluate(TXT_DIR_OUT_,file_name)
        print('=========================\n') if(DEBUG) else None
    print('----------------------------------------')
    print('Evaluation Completed!')
    return None


def main():
    # main function
    
    evaluateModelOutput("output_1/")
    evaluateModelOutput("output_2/")
    evaluateModelOutput("output_3/")
    evaluateModelOutput("output_4/")
    evaluateModelOutput("output_5/")
    evaluateModelOutput("output_keylug/")
    
    return None
    
main()

