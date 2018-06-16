# -*- coding: utf-8 -*-
""" 
Model 4: GraphicalCollapseKE
Author : Vidhya Moorthy
"""

import operator
import glob
import nltk
import os
from nltk.corpus import stopwords
import re
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import wordnet
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
import matplotlib.pyplot as plt
import networkx as nx
import math
from itertools import product

wscores = []
tempdict = {}
sorteddict = {}
predict=[]
keyword_list=[]
# sets the current directory
os.chdir("C:/Users/Vidhya Moorthy/Desktop/Hybrid Keyword Extraction")

# prints the current working directory
print(os.getcwd())

# prints the directory contents
os.listdir('.')

# global variables
DEBUG=False # true when debugging to display print messages

TXT_DIR_IN="input/"   # input  - contains .txt files with raw text
TXT_DIR_OUT="output_4/" # output - contains .out files with predicted values

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
    
# removes punctuation
def removePunctuation(text):
    filtered = re.sub(r'[^A-Za-z0-9\s]+', '', text).strip().lower()
    return filtered
    
    
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


	
	
# returns semantic score   
def getSemanticScore(word1,word2):
    word1 = word1+'.n.01'
    word2 = word2+'.n.01'
    ws1 = wordnet.synset(word1)
    ws2 = wordnet.synset(word2)
    return ws1.wup_similarity(ws2)    

# drawing weighted graph
# creating an empty graph
G=nx.Graph() # graph before collapse
G_result = nx.Graph() # graph after collapse


def constructGraph_beforeCollapse(threshold):
    elarge = [(u,v) for (u,v,d) in G_result.edges(data=True) if d['weight'] >=threshold]
    esmall = [(u,v) for (u,v,d) in G_result.edges(data=True) if d['weight'] <threshold]
    
#G.remove_edges_from(esmall)

# positions for all nodes
    pos = nx.spring_layout(G_result,scale=10000000000^100000000000000) 
# NOTE: run the commands below together to get graph with all the features
# nodes
    nx.draw_networkx_nodes(G_result,pos,node_size=350)
# edges
    nx.draw_networkx_edges(G_result,pos,edgelist=elarge,width=2,alpha=0.5,edge_color='black')
# node labels
    nx.draw_networkx_labels(G_result,pos,font_size=10,font_family='sans-serif')
# edge labels
    labels = nx.get_edge_attributes(G_result,'weight')
    nx.draw_networkx_edge_labels(G_result,pos,edge_labels=labels,font_size=5)    

    plt.axis('off')
    plt.savefig("weighted_graph_afterCollapse.png")
    plt.show() # display
	
def constructGraph_afterCollapse(threshold):
    elarge = [(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] >=threshold]
    esmall = [(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] <threshold]
    
#G.remove_edges_from(esmall)

# positions for all nodes
    pos = nx.spring_layout(G,scale=10000000000^100000000000000) 
# NOTE: run the commands below together to get graph with all the features
# nodes
    nx.draw_networkx_nodes(G,pos,node_size=350)
# edges
    nx.draw_networkx_edges(G,pos,edgelist=elarge,width=2,alpha=0.5,edge_color='black')
# node labels
    nx.draw_networkx_labels(G,pos,font_size=10,font_family='sans-serif')
# edge labels
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G,pos,edge_labels=labels,font_size=5)    

    plt.axis('off')
    plt.savefig("weighted_graph_beforeCollapse.png")
    plt.show() # display
	
# calculates semantic scores for data sets	
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
    nouns_only=[]
    processed_nouns = []
    f_list =[]
    wscores = []
    tempdict = {}
    sorteddict = {}
    predict=[]
    keyword_list=[]
    


    f = open(in_file_path_,mode='r',encoding="utf-8")
    doc = f.read()

    # tokenization
    tokens = nltk.word_tokenize(doc)
    print(tokens) if(DEBUG) else None

    # stop word removal
    filtered_words = [word for word in tokens if word not in stopwords.words('english')]
    print(filtered_words) if(DEBUG) else None

    # parts of speech tagging for filtered words
    tagged_filtered_words = nltk.pos_tag(filtered_words)
    print(tagged_filtered_words) if(DEBUG) else None

    # extracting only nouns and verbs from filtered words
    for word,tag in tagged_filtered_words:
        if(tag == 'NN' or tag == 'NNP' or tag == 'NNS' or tag == 'NNPS'):
            filtered_nouns.append(word)
    # extracting named entities from filtered_nouns list
    ne_nouns = nltk.ne_chunk(nltk.pos_tag(filtered_nouns), binary=True)
    
    for tree in ne_nouns:
        # Print results per sentence
        # print extract_entity_names(tree)
        entity_names.extend(extract_entity_names(tree))
    # Print unique entity names
    print(entity_names) if(DEBUG) else None
    #remove the entity names from filtered_nouns
    non_entityList = [entity for entity in filtered_nouns if entity not in entity_names]
    print(non_entityList) if(DEBUG) else None

    # removing punctuation
    for word in filtered_nouns:
        processed.append(removePunctuation(word).replace(" ", "_"))

    # remove empty words
    for word in processed:
        if not word:
            processed.remove(word)    

    # removing invalid words in WordNet
    wpt = WordPunctTokenizer()

    for s in processed:
        tokens = wpt.tokenize(s)
        if tokens:  # check if empty string
            for t in tokens:
                if wordnet.synsets(t):
                    processed_valid_words.append(t)  # only keep recognized words

    print(processed_valid_words) if(DEBUG) else None

    wordtags = nltk.ConditionalFreqDist((w.lower(), t)
        for w, t in nltk.corpus.brown.tagged_words(tagset="universal")) 


    for word1 in processed_valid_words:
        print(word1,list(wordtags[word1])) if(DEBUG) else None
        if 'NOUN' in list(wordtags[word1]):
            nouns_only.append(word1)

    processed_valid_words = nouns_only

    #part of speech tagging for processed words so that it can be checked for noun
    tagged_processed_words = nltk.pos_tag(processed_valid_words)

    # extracting only nouns from processed words
    for word,tag in tagged_processed_words:
        if(tag == 'NN' or tag == 'NNP' or tag == 'NNS' or tag == 'NNPS'):
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
        
    # traversing the list of filtered nouns to calculate the semantic score using wordnet
    print("Before Collapse") if(DEBUG) else None
    for word1, word2 in product(final_list,final_list):
        if word1 != word2:
            wscore = getSemanticScore(word1,word2)
            #ceiling the long value
            wscore = math.ceil(wscore*100)/100
            wscores.append(wscore) # adding the wscores to a list
            G.add_edge(word1,word2,weight=wscore)
            print(word1," ",word2,"->",wscore) if(DEBUG) else None

    constructGraph_beforeCollapse(0)
    max_score = max(wscores)
    threshold = max_score/2

    print("After Collapse") if(DEBUG) else None
    # collapsing the graph
    for word1, word2 in product(final_list,final_list):
        if word1 != word2:
            wscore = getSemanticScore(word1,word2)
            #ceiling the long value
            wscore = math.ceil(wscore*100)/100
            if wscore > threshold:
                G_result.add_edge(word1,word2,weight=wscore)
                lst=[word1,word2]  #print(word1," ",word2,"->",wscore)
                label=lst[0]+" <--> "+lst[1]
                if label not in tempdict:   # To update the label and  
                    tempdict.update({label:wscore})
                else:
                    tempdict[label]=tempdict[label]+wscore
                 
    constructGraph_afterCollapse(threshold)                    
    sorteddict = sorted(tempdict.items(),key=operator.itemgetter(1),reverse=True)
    for item in sorteddict[:20]:
       test=item[0].split("<-->")
       predict.append(test[0].strip()+" "+test[1].strip())
       #print("Keywords: "+test[0].strip()+" "+test[1].strip()+" Force: "+str(item[1]))
       print("Keywords: "+test[0].strip()+" "+test[1].strip()) if(DEBUG) else None
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
    print('\nExtracting Keywords using Graphical Model:')
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
    print('\nOutput files (.out) saved in directory output_4/')
    return None
    
main()