import numpy as np
import re
import itertools
from collections import Counter

"""
Original taken from https://github.com/dennybritz/cnn-text-classification-tf
"""
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

from os import listdir
import sys

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def harm_mean(a, b):
    if a*b == 0.0:
        return 0.0
    return 2*a*b/(a+b)

def getRunData(runNo):

  datasets = {}
  all_para_same = False

  trainingSet1 = [0,1,3,4,6,7,10,11,12,13,14,16,17,18,23,24,28,29,30,31,33,36,38]
  devSet1 = [27,32,15,25,5,8,22,21]
  testSet1 = [35,9,20,34,37,26,19,2]
  SVC_para_1 = -1
  NB_para_1 = -1

  trainingSet2 = [1,2,5,6,8,13,15,16,17,18,19,20,22,23,25,28,30,31,34,35,36,37,38]
  devSet2 = [33,7,14,26,11,4,32,12]
  testSet2 = [21,24,0,29,27,3,9,10]
  SVC_para_2 = -1
  NB_para_2 = -1

  trainingSet3 = [0,1,3,5,7,9,10,11,12,13,16,18,19,21,23,24,25,26,29,32,35,36,38]
  devSet3 = [20,17,4,31,34,14,2,37]
  testSet3 = [27,15,6,22,33,8,30,28]
  SVC_para_3 = -2
  NB_para_3 = 0
  if all_para_same:
    SVC_para_3 = -1
    NB_para_3 = -1

  trainingSet4 = [2,3,5,6,8,9,10,11,13,14,16,17,19,20,22,23,27,29,30,31,32,34,38]
  devSet4 = [1,4,35,26,33,25,12,0]
  testSet4 = [18,15,36,37,7,28,21,24]
  SVC_para_4 = 1
  NB_para_4 = 0
  if all_para_same:
    SVC_para_4 = -1
    NB_para_4 = -1

  trainingSet5 = [1,2,7,8,11,12,13,14,15,16,19,20,22,23,24,26,27,29,31,32,33,36,38]
  devSet5 = [4,30,17,18,0,6,37,35]
  testSet5 = [21,3,28,25,34,10,5,9]
  SVC_para_5 = -1
  NB_para_5 = 0
  if all_para_same:
    SVC_para_5 = -1
    NB_para_5 = -1

  datasets[1] = (trainingSet1,devSet1,testSet1,SVC_para_1,NB_para_1)
  datasets[2] = (trainingSet2,devSet2,testSet2,SVC_para_2,NB_para_2)
  datasets[3] = (trainingSet3,devSet3,testSet3,SVC_para_3,NB_para_3)
  datasets[4] = (trainingSet4,devSet4,testSet4,SVC_para_4,NB_para_4)
  datasets[5] = (trainingSet5,devSet5,testSet5,SVC_para_5,NB_para_5)

  return datasets[runNo]

def generateData(fileIndex,dataFolder):

  X = []
  Y = []
  sentence = ''
  relevance = 0

  for i,fileName in enumerate(listdir(dataFolder)):
    if i in fileIndex:

      with open(dataFolder+fileName,'r') as f:
        for line in f:
          if line=='\n':
            if sentence!='':
              X.append(sentence)
              Y.append(relevance)
            sentence = ''
            relevance = 0
          else:
            if sentence=='': sentence = line.split(' ')[0]
            else: sentence+=' '+line.split(' ')[0]
            if line[:-1].split(' ')[-1]!='O': relevance = 1

  return X,Y


def load_data_and_labels(run):
        """
        Loads MR polarity data from files, splits the data into words and generates labels.
        Returns split sentences and labels.
        """
        # Load data from files
        tokenizedFolder = "D:/malwareTextAnalysis/data/tokenized/"

        trainingSet, devSet, testSet, SVC_para, NB_para = getRunData(run)

        trainX, trainY = generateData(trainingSet, tokenizedFolder)
        devX, devY = generateData(devSet, tokenizedFolder)
        testX, testY = generateData(testSet, tokenizedFolder)

        trainX=[clean_str(sent) for sent in trainX]
        trainX=[s.split(" ") for s in trainX]
        trainYY=[]
        testYY=[]
        devYY=[]
        for i in trainY:
            if i==0:
                trainYY.append([0,1])
            else:
                trainYY.append([1,0])
        devx=[clean_str(sent) for sent in devX]
        devX = [s.split(" ") for s in devX]
        for i in devY:
            if i==0:
                devYY.append([0,1])
            else:
                devYY.append([1,0])
        testX=[clean_str(sent) for sent in testX]
        testX = [s.split(" ") for s in testX]
        for i in testY:
            if i==0:
                testYY.append([0,1])
            else:
                testYY.append([1,0])

        return(trainX,trainYY,devX,devYY,testX,testYY)



def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def load_data(run):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentencestrain, labelstrain,sentencesdev,labelsdev,sentencestest,labelstest = load_data_and_labels(run)
    sentences,labels=sentencestrain+sentencesdev+sentencestest,labelstrain+labelsdev+labelstest
    train_len=len(sentencestrain)
    devlen=len(sentencesdev)
    testlen=len(sentencestest)
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv,train_len,devlen,testlen]


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
