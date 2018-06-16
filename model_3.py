# -*- coding: utf-8 -*-
"""
Model 3: hybrid_GraphPP_modified_algo
Author : Ashwini Belgundkar
"""

from __future__ import division
import os
import sys
import glob
import operator
import time
import math
import numpy.random as np
import matplotlib.pyplot as plt
import networkx as nx
import re

import nltk
from nltk.corpus import wordnet,stopwords
from nltk.corpus import wordnet as wn
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize

from itertools import islice, product
from collections import Counter


# sets the current directory
os.chdir("C:/Users/Vidhya Moorthy/Desktop/Hybrid Keyword Extraction")

# prints the current working directory
print(os.getcwd())

# prints the directory contents
os.listdir('.')

# global variables
DEBUG=False # true when debugging to display print messages

TXT_DIR_IN="input/"   # input  - contains .txt files with raw text
TXT_DIR_OUT="output_3/" # output - contains .out files with predicted values


# extracts entity names    
def extract_entity_names(t):
    entity_names = []
    if hasattr(t, 'label') and t.label:
        if t.label() == 'NE':
            entity_names.append(' '.join([child[0] for child in t]))
        else:
            for child in t:
                entity_names.extend(extract_entity_names(child))

    return entity_names

# returns true if string contains number
def hasNum(strng):
    return any(char.isdigit() for char in strng)

# removes punctuation
def removePunctuation(text):
    filtered = re.sub(r'[^A-Za-z0-9\s]+', '', text).strip().lower()
    return filtered

# returns semantic score
def getSemanticScore(word1,word2):
    word1 = word1+'.n.01'
    word2 = word2+'.n.01'
    ws1 = wordnet.synset(word1)
    ws2 = wordnet.synset(word2)
    return ws1.wup_similarity(ws2)


def penn_to_wn(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None


def findAttraction(file_name_string):

    # generating paths
    # ----------------------------------------    
    in_file_path_ = TXT_DIR_IN + file_name_string + '.txt'
    out_file_path_ = TXT_DIR_OUT + file_name_string + '.out'
    
    # local variables
    # ----------------------------------------  
    filtered_nouns = []
    entity_names = []
    processed = []
    processed_valid_words = []
    processed_nouns = []
    nouns_only=[]
    f_list =[]
    
    wWeight={}
    wPos={}
    attraction={}
    termfreq={}
    tmpDict={}
    totaltermfrequency={}
    stpwds=[]
    keys=[]
    keyword_list=[]
    
    numOfWords=0
    
    # reading file
    f = open(in_file_path_,mode='r',encoding="utf-8")
    doc = f.read()
    
    # tokenization
    tokens = nltk.word_tokenize(doc)
    print(tokens) if(DEBUG) else None
    
    # removing stopwords
    filtered_words = [word for word in tokens if word not in stopwords.words('english')]
    print(filtered_words) if(DEBUG) else None
    
    # parts of speech tagging for filtered words
    tagged_filtered_words = nltk.pos_tag(filtered_words)
    print(tagged_filtered_words) if(DEBUG) else None
    
    # extracting only nouns from filtered words
    for word,pos_tag in tagged_filtered_words:
        if(pos_tag == 'NN' or pos_tag == 'NNP' or pos_tag == 'NNS' or pos_tag == 'NNPS'):
            filtered_nouns.append(word)
        #if(pos_tag == 'VB' or pos_tag == 'VBD' or pos_tag == 'VBG' or pos_tag == 'VBN' or pos_tag =='VBP' or pos_tag == 'VBZ'):
         #   filtered_verbs.append(word)
            
    ne_nouns = nltk.ne_chunk(nltk.pos_tag(filtered_nouns), binary=True)
    
    for tree in ne_nouns:
        # Print results per sentence
        # print extract_entity_names(tree)

        entity_names.extend(extract_entity_names(tree))

    # Print unique entity names
    print(entity_names) if(DEBUG) else None

    # remove the entity names from filtered_nouns
    non_entityList = [entity for entity in filtered_nouns if entity not in entity_names]
    print(non_entityList) if(DEBUG) else None

    # remove punctuations
    for word in non_entityList:
        processed.append(removePunctuation(word).replace(" ", "_"))

    #remove empty words
    for word in processed:
        if not word:
            processed.remove(word)    
    print(processed) if(DEBUG) else None


    wpt = WordPunctTokenizer()

    for s in processed:
        tokens = wpt.tokenize(s)
        if tokens:  # check if empty string
            for t in tokens:
                if wordnet.synsets(t):
                    processed_valid_words.append(t)  # only keep recognized words

    print(processed_valid_words) if(DEBUG) else None

    
    # removing words that are not nouns from processed_valid_words    
    wordtags = nltk.ConditionalFreqDist((w.lower(), t) 
    for w, t in nltk.corpus.brown.tagged_words(tagset="universal"))

    print("------------")# if(DEBUG) else None
    for word1 in processed_valid_words:
        print(word1,list(wordtags[word1]))# if(DEBUG) else None
        if 'NOUN' in list(wordtags[word1]):
            nouns_only.append(word1) 

    print("------------")# if(DEBUG) else None
    for word in nouns_only:
        print(word,list(wordtags[word]))# if(DEBUG) else None
        
    processed_valid_words = nouns_only
    

    #part of speech tagging for processed words so that it can be checked for noun
    tagged_processed_words = nltk.pos_tag(processed_valid_words)
    
    
    # extracting only nouns from processed words
    for word,pos_tag in tagged_processed_words:
        if(pos_tag == 'NN' or pos_tag == 'NNP' or pos_tag == 'NNS' or pos_tag == 'NNPS'):
            processed_nouns.append(word)

    tagged_processed_nouns = nltk.pos_tag(processed_nouns)
    final_noun_set = [item[0] for item in tagged_processed_nouns if item[1][0] == 'N']
    final_noun_tagged = nltk.pos_tag(final_noun_set)   
    print(final_noun_set) if(DEBUG) else None

    # Lemmatization -- > removal of plurals 
    lemmatzr = WordNetLemmatizer()
    

    for noun in final_noun_tagged:
        wn_tag = penn_to_wn(noun[1])
        if not wn_tag:
            continue

        lemma = lemmatzr.lemmatize(noun[0], pos=wn_tag)
        f_list.append(lemma)

    # removing duplicates in the f_list by converting to a set  
    f_set = set(f_list)
    final_list = list(f_set)
    print(final_list) if(DEBUG) else None
    

    #print (final_list)
    for word in final_list:
        numOfWords=numOfWords+1
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
                    
    
    #print(numOfWords)
    length=int(numOfWords/10)     # The sliding window is determined by lambda = length = (numofwords/100) , this can be modified 
    
    for j in range(1, numOfWords):  # j will range from 1 to length 
        # Looking for attraction within specified range. (1% of words)
        for i in range(j-length,j+length):  # i will range from j-length to j+length 
            if i in wPos and j in wPos and j!=i:  
                #totaltermfrequency = termfreq[i]*termfreq[j]
                #print (totaltermfrequency)
                if wPos[j] != wPos[i]:    
                    #force=(wWeight[wPos[j]]*wWeight[wPos[i]]) 
                    force = getSemanticScore(wPos[i],wPos[j])
                    #print (force)
                    force=force/(abs(j-i)**2)
                    lst=[wPos[i],wPos[j]]
                    lst.sort()
                    label=lst[0]+" <--> "+lst[1]
                    if label not in tmpDict:   # To update the label and force 
                        tmpDict.update({label:force})
                    else:
                        tmpDict[label]=tmpDict[label]+force 
                        
    outDict = sorted(tmpDict.items(),key=operator.itemgetter(1),reverse=True) 
    # operator function, which lists all the items , sorts according to 2nd item , checks if the reverese is true
    #print (outDict[1:5])
    predict=[] # ashwini: defining globally
    gp=True # ashwini: defining globally
    size=len(outDict)
    print("Relationships: "+str(size)) if(DEBUG) else None
    #print (outDict[1:5])
    
    for item in outDict[:10]:
        test=item[0].split("<-->")
        predict.append(test[0].strip()+" "+test[1].strip())
        print("Keywords: "+test[0].strip()+" "+test[1].strip()+" Force: "+str(item[1])) if(DEBUG) else None
        keyword_list.append(test[0].strip()) # abhi: for evaluation
        keyword_list.append(test[1].strip()) # abhi: for evaluation
    
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
    print('\nExtracting Keywords using Model 3:')
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
    print('\nOutput files (.out) saved in directory output_3/')
    return None
    
main()






