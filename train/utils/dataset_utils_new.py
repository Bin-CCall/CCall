# -*- coding: utf-8 -*-
# @Time    : 2021/7/24 下午1:34
# @Author  : Dai Pu wei
# @Email   : 771830171@qq.com
# @File    : dataset_utils.py
# @Software: PyCharm

import numpy as np
import pickle
import gc
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences
from  sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
import time
from tensorflow.keras.utils import to_categorical

def tokenize(data_pre):
    with open("./word2id_nopad.pkl","rb") as w2id_f:
        w2id = pickle.load(w2id_f)
    data = []
    for d in data_pre:
        d_new =  [w2id[w] for w in d if w in w2id.keys()]  
        #if(d_new in data):
            #print(d_new)
        data.append(d_new)
        '''
        if(len(d_new)!=len(d)):
            print(d_new)
            print(d)
        '''
        #print((len(d_new),len(d)))
    assert(len(data)==len(data_pre))
    return data

def preprocess_dataset(data_dir,seq_len,p):
    
    with open(data_dir, 'rb') as pos:
        positive_data = tokenize(pickle.load(pos))

    with open(data_dir.replace("positive","negative"),'rb') as neg:
        negative_data = tokenize(pickle.load(neg))
    print('loading data from {data_dir}...'.format(data_dir=data_dir))
    
    x_pre = positive_data + negative_data
    y_positive_labels = [1 for _ in positive_data]
    y_negative_labels = [0 for _ in negative_data]
    y = y_positive_labels + y_negative_labels
    del positive_data
    del negative_data
    gc.collect()
    x_pre = pad_sequences(x_pre, maxlen=int(seq_len) + 1, dtype='int32', padding='post', truncating='post', value=0)
    assert(len(x_pre)==len(y))
    x = np.asarray(x_pre)
    y = np.asarray(y)
    
    return x,y

def split_dataset_noneg(data_dir,seq_len,p):

    with open(data_dir, 'rb') as pos:
        positive_data_pre = pickle.load(pos)

    num = int(p*len(positive_data_pre))

    
    print('loading data from {data_dir}...'.format(data_dir=data_dir))
    positive_index_slice = np.asarray(random.sample([x for x in range(len(positive_data_pre))],num),dtype=np.int32)
    positive_data = np.asarray(positive_data_pre)[positive_index_slice]
    positive_data = tokenize(positive_data.tolist())
    del positive_data_pre
    gc.collect()
    
    x_pre = positive_data
    y_positive_labels = [1 for _ in positive_data]
    
    y = y_positive_labels
    del positive_data
    
    gc.collect()
    x_pre = pad_sequences(x_pre, maxlen=int(seq_len) + 1, dtype='int32', padding='post', truncating='post', value=0)
    assert(len(x_pre)==len(y))
    x_train = np.asarray(x_pre)[:int(0.8*len(x_pre))]
    y_train = np.asarray(y)[:int(0.8*len(x_pre))]
    x_val = np.asarray(x_pre)[int(0.8*len(x_pre)):]
    y_val = np.asarray(y)[int(0.8*len(x_pre)):]
    
    return x_train,y_train,x_val,y_val

def split_dataset(data_dir,seq_len,p):

    with open(data_dir, 'rb') as pos:
        positive_data_pre = pickle.load(pos)

    with open(data_dir.replace("positive","negative"),'rb') as neg:
        negative_data_pre = pickle.load(neg)
        
    num = int(min(p*len(positive_data_pre),p*len(negative_data_pre)))

    
    print('loading data from {data_dir}...'.format(data_dir=data_dir))
    positive_index_slice = np.asarray(random.sample([x for x in range(len(positive_data_pre))],num),dtype=np.int32)
    positive_data = np.asarray(positive_data_pre)[positive_index_slice]
    positive_data = tokenize(positive_data.tolist())
    del positive_data_pre
    gc.collect()

    print('loading data from {data_dir}...'.format(data_dir=data_dir.replace("positive","negative")))
    #negative_data = []
    negative_index_slice = np.asarray(random.sample([x for x in range(len(negative_data_pre))],num),dtype=np.int32)
    
    negative_data = np.asarray(negative_data_pre)[negative_index_slice]
    negative_data = tokenize(negative_data.tolist())
    del negative_data_pre
    gc.collect()

    x_pre = positive_data + negative_data
    y_positive_labels = [1 for _ in positive_data]
    y_negative_labels = [0 for _ in negative_data]
    y = y_positive_labels + y_negative_labels
    del positive_data
    del negative_data
    gc.collect()
    x_pre = pad_sequences(x_pre, maxlen=int(seq_len) + 1, dtype='int32', padding='post', truncating='post', value=0)
    assert(len(x_pre)==len(y))

    ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, train_size=0.8,random_state=1154)
    for test_index, fake_index in ss.split(x_pre, y):
        x_train, x_val = np.asarray(x_pre)[test_index], np.asarray(x_pre)[fake_index]#训练集对应的值
        y_train, y_val = np.asarray(y)[test_index], np.asarray(y)[fake_index]#类别集对应的值
        #print(Counter(y_train.tolist()))
        #print(Counter(y_val.tolist()))
    
    return x_train,y_train,x_val,y_val


def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    num = data[0].shape[0]
    p = np.random.permutation(num)
    #两个打乱的一维list x和y打乱的一致
    return [d[p] for d in data]

def batch_generator(data, batch_size, shuffle=True):
    """Generate batches of data.

    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
    if shuffle:
        data = shuffle_aligned_list(data)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data[0]):
            batch_count = 0

            if shuffle:
                data = shuffle_aligned_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]