from sys import *
import os
from os.path import *
from numpy import *
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
import gzip
from stopwords import list_stopwords
import numpy as np
import xml.etree.ElementTree as ET
import re
from pattern.en import lemma
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
def main(inputFileDir):
    '''
    inputFilePath = join(inputFileDir, 'rr', 'ManuallyAnnotated_Corpus.txt')
    outputFilePath = join(inputFileDir, 'rr', 'pro_ManuallyAnnotated_Corpus.txt')
    inputFile = open(inputFilePath, 'r')
    outputFile = open(outputFilePath, 'w')

    digitkey = re.compile('<\d+>|</\d+>|xmlns=\".*\"')
    for line in inputFile:
        if line == '':
            continue
        if line.startswith('Restaurant Name'):
            continue

        line_new = re.sub(digitkey, ' ', line)
        outputFile.write(line_new)
    outputFile.close()
    
    inputFile.close()
    '''
    #print (outputFilePath)
    outputFilePath = '/Users/cheryl/summer/codes/preprocess/rr/pro_Classified_Corpus.txt'
    tree = ET.parse(outputFilePath)
    root = tree.getroot()
    print (root)
    rrTestPath = join(inputFileDir, 'rr', 'test_rr.txt')
    rrTest = open(rrTestPath, 'w')
    
    #stemmer = PorterStemmer()
    stemmer = WordNetLemmatizer()
    tokenizer = RegexpTokenizer(r'\w+')
    
    num = 0
    for restaurant in root:
        for review in restaurant:
            
            if len(review) > 50:
                break
            
            for doc in review:
                
                taglist = []
                leaf_elem = []
                for elem in doc.iter():
                    tag_temp = elem.tag
                    if tag_temp not in ['Positive','Negative','Neutral']:
                        taglist.append(elem.tag)
                    leaf_elem = elem
                tagnum = len(taglist)
                
                # doc text
                docline = leaf_elem.text
                #print (docline)
                doctext = docline.lower()
                #print (doctext)
                docwords = tokenizer.tokenize(doctext)
                #print(docwords)
                sWordList = []
                for docword,pos in nltk.pos_tag(docwords):
                    if pos.startswith("N") or pos.startswith("V"):
                        docword = lemma(docword)
                    print(docword)
                    #if reviewWord.isalpha() and (reviewWord not in list_stopwords):
                    if docword.isalpha() and docword not in list_stopwords:
                        sWordList.append(docword)
                    
                if tagnum != 1 or sWordList == []:
                    continue
                rrTest.write(str(num)+'\t')
                '''
                for tagstr in taglist:
                    rrTest.write(tagstr+'\t')
                '''
                for word in sWordList:
                    rrTest.write(word+' ')
                rrTest.write('\n')
            num += 1
            
    print(num)
    rrTest.close()

        
if __name__ == '__main__':
    main(argv[1])