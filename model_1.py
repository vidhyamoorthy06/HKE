# -*- coding: utf-8 -*-
""" 
Model 1: statistical_weighted.py 
Author : Ashwini Belgundkar
"""

from __future__ import division
import os
import sys
import glob
import operator
import time
import numpy.random as np
import matplotlib.pyplot as plt
import networkx as nx

# sets the current directory
os.chdir("C:/Users/Vidhya Moorthy/Desktop/Hybrid Keyword Extraction")

# prints the current working directory
print(os.getcwd())

# prints the directory contents
os.listdir('.')

# global variables
DEBUG=False # true when debugging to display print messages

TXT_DIR_IN="input/"   # input  - contains .txt files with raw text
TXT_DIR_OUT="output_1/" # output - contains .out files with predicted values


# returns true if string contains number
def hasNum(strng):
    return any(char.isdigit() for char in strng)

    
# assigns weights to all words in file and calculates relatedness between two words
def findAttraction(file_name_string):
    
    # generating paths
    # ----------------------------------------    
    in_file_path_ = TXT_DIR_IN + file_name_string + '.txt'
    out_file_path_ = TXT_DIR_OUT + file_name_string + '.out'
    
    # local variables
    # ----------------------------------------    
    outDict={}
    termfreq={}    
    tmpDict={}
    wPos={}
    wWeight={}

    doc=[]
    keyword_list=[]
    predict=[]
    stpwds=[]
    test=[]

    numOfWords=0
    gp=True
    
    
    with open(in_file_path_,mode='r',encoding="utf-8") as stops:
        for line in stops:
            stopwds=line.split()
            for wds in stopwds:
                stpwds.append(wds)            

    with open(in_file_path_,mode='r',encoding="utf-8") as f:
        for line in f:
            doc.append(line)
            words=line.split()
            for word in words:
                numOfWords=numOfWords+1
                word=''.join(e for e in word if e.isalnum()) # Joins words in a row with a space in between 
                word=word.lower() # converts word to lower case
                if word not in stpwds:  # Checks if not in stopword list
                    if word not in termfreq:  # if not in termfreq list , then checks if the length of word is more than 2, and updates #1 to that word
                        if len(word) > 2:
                            termfreq.update({word:1}) # Will update the word as digit 1
                    if word in termfreq: # if its there in termfreq , the termfreq [word] for that word will get incremented
                        if len(word) > 2: #Both cases it checks for the length of the word to be more than 2
                            termfreq[word]=termfreq[word]+1 # If word is repeating , this will increment the count
                            
                    if word not in wPos:     # Sets the position of the word for a new word 
                        if len(word) > 2:
                            wPos.update({numOfWords:word})  # Dictionary key-value 

                    if word not in wWeight:  # Sets the weight of the word to zero 
                        if len(word) > 2:
                            wWeight.update({word:0})

                    if word in wWeight:
                        if len(word) > 2:
                            if not hasNum(word):    #checks if the word has number in it , if not weight of the word = len (word)
                                wWeight[word]=len(word)+wWeight[word]+termfreq[word]
                                
    start=time.time()
    length= numOfWords    # The sliding window is determined by lambda = length = (numofwords/100) , this can be modified 

    for j in range(1, numOfWords): # j will range from 1 to length 
    
        # Looking for attraction within specified range. (1% of words)
        for i in range(j-length,j+length):  # i will range from j-length to j+length 
            if i in wPos and j in wPos and j!=i:  
                
                ###sys.stdout.write("Progress: "+"{0:.2f}".format(100*j/numOfWords)+"%\r")
                #print("Progress: "+"{0:.2f}".format(100*j/numOfWords)+"%\r")
                
                #totaltermfrequency = termfreq[i]*termfreq[j]
                #print (totaltermfrequency)
                if wPos[j] != wPos[i]:    
                    force=(wWeight[wPos[j]]*wWeight[wPos[i]]) 
                    force=force/(abs(j-i)**2)
                    lst=[wPos[i],wPos[j]]
                    lst.sort()
                    label=lst[0]+" <--> "+lst[1]
                    if label not in tmpDict:   # To update the label and force 
                        tmpDict.update({label:force})
                    else:
                        tmpDict[label]=tmpDict[label]+force
    
    #global outDict
    outDict = sorted(tmpDict.items(),key=operator.itemgetter(1),reverse=True) 
    # operator function, which lists all the items , sorts according to 2nd item , checks if the reverese is true
    
    stop=time.time()

    size=len(outDict)
    print("\nRelationships: "+str(size)) if(DEBUG) else None
    #global predict

    for item in outDict:
        if len(predict) <10:
            #global test
            test=item[0].split("<-->")
            for thing in predict:
                if test[0].strip() in thing or test[1].strip()  in thing:
                    gp=False
                    break;
                else:
                    gp=True
            if gp:
                predict.append(test[0].strip()+" "+test[1].strip())
                print("NP: "+test[0].strip()+" "+test[1].strip()+" Force[N]: "+str(item[1])) if(DEBUG) else None
                keyword_list.append(test[0].strip()) # abhi: for evaluation
                keyword_list.append(test[1].strip()) # abhi: for evaluation
                
        else:
            break
    print("\n>> Completed @ "+str(stop-start)) if(DEBUG) else None
    
    
    # abhi: generating .out file
    # ----------------------------------------
    out = open(out_file_path_,mode="w",encoding="utf-8")
    for keyword in keyword_list:
        print(keyword) if(DEBUG) else None
        print(keyword, file=out) # abhi: each keyword will be appended to the file in a new line
    out.close()
    print("Done.")
    return None

def main():    
    # main function    
    print('\nExtracting Keywords using Model 1:')
    print('----------------------------------------')
    # getting a list of files
    file_list = glob.glob(TXT_DIR_IN+"*.txt")
    
    # iterate over the list getting each file 
    for file_name in file_list:
        file_name = file_name[:-4] # removing file extention suffix
        file_name = file_name.replace(TXT_DIR_IN[:-1]+'\\','') # removing dir prefix
        print(file_name,'... ', end='')
        findAttraction(file_name)
        print('=========================\n') if(DEBUG) else None
    print('----------------------------------------')
    print('Keyword Extraction Completed!')
    print('\nOutput files (.out) saved in directory output_1/')
    return None
    
main()