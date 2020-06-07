#!/usr/bin/env python
# coding: utf-8

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import click as ck
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import function
import re
import math
import matplotlib.pyplot as plt
import logging
from tensorflow.keras.layers import (
    Input,
)
from tensorflow.keras import optimizers
from tensorflow.keras import constraints
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras import backend as K
from scipy.stats import rankdata
from tensorflow.python.keras.utils.data_utils import Sequence


config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

logging.basicConfig(level=logging.INFO)

def load_valid_data(valid_data_file, classes, relations):
    data = []
    rel = f'SubClassOf'
    with open(valid_data_file, 'r') as f:
        for line in f:
            it = line.strip().split()
            id1 = it[0]
            id2 = it[1]
            if id1 not in classes or id2 not in classes or rel not in relations:
                continue
            data.append((classes[id1], relations[rel], classes[id2]))
    return data

def load_cls(train_data_file):
    train_subs=list()
    counter=0
    with open(train_data_file,'r') as f:
        for line in f:
            counter+=1
            it = line.strip().split()
            cls1 = it[0]
            cls2 = it[1]
            train_subs.append(cls1)
            train_subs.append(cls2)
    train_cls = list(set(train_subs))
    return train_cls,counter

total_sub_cls=[]
train_file = "NCI_data/NCI_train.txt"
va_file = "NCI_data/NCI_valid.txt"
test_file = "NCI_data/NCI_test.txt"
train_sub_cls,train_samples = load_cls(train_file)
valid_sub_cls,valid_samples = load_cls(va_file)
test_sub_cls,test_samples = load_cls(test_file)
total_sub_cls = train_sub_cls + valid_sub_cls + test_sub_cls

all_subcls = list(set(total_sub_cls))

print("Training data samples:",train_samples)
print("Training data classes:",len(train_sub_cls))

#Original Loss
#Gallen no disjoints and nf2
def load_data(filename):
    classes = {}
    relations = {}
    data = {'nf1': [], 'nf3': [], 'nf4': []}
    with open(filename) as f:
        for line in f:
            if line.startswith('FunctionalObjectProperty'):
                continue
            # Ignore SubObjectPropertyOf
            if line.startswith('SubObjectPropertyOf'):
                continue
            # Ignore SubClassOf()
            line = line.strip()[11:-1]
            if not line:
                continue
            if line.startswith('ObjectIntersectionOf('):
                continue
                
            elif line.startswith('ObjectSomeValuesFrom('):
                # R some C SubClassOf D
                it = line.split(' ')
                r = it[0][21:]
                c = it[1][:-1]
                d = it[2]
                if c not in classes:
                    classes[c] = len(classes)
                if d not in classes:
                    classes[d] = len(classes)
                if r not in relations:
                    relations[r] = len(relations)
                data['nf4'].append((relations[r], classes[c], classes[d]))
            elif line.find('ObjectSomeValuesFrom') != -1:
                # C SubClassOf R some D
                it = line.split(' ')
                c = it[0]
                r = it[1][21:]
                d = it[2][:-1]
                if c not in classes:
                    classes[c] = len(classes)
                if d not in classes:
                    classes[d] = len(classes)
                if r not in relations:
                    relations[r] = len(relations)
                data['nf3'].append((classes[c], relations[r], classes[d]))
            else:
                # C SubClassOf D
                it = line.split(' ')
                c = it[0]
                d = it[1]
                r = 'SubClassOf'
                if r not in relations:
                    relations[r] = len(relations)
                if c not in classes:
                    classes[c] = len(classes)
                if d not in classes:
                    classes[d] = len(classes)
                data['nf1'].append((classes[c],relations[r],classes[d]))
                
    # Check if TOP in classes and insert if it is not there
    if 'owl:Thing' not in classes:
        classes['owl:Thing'] = len(classes)
#changing by adding sub classes of train_data ids to prot_ids
    prot_ids = []
    class_keys = list(classes.keys())
    for val in all_subcls:
        if val not in class_keys:
            cid = len(classes)
            classes[val] = cid
            prot_ids.append(cid)
        else:
            prot_ids.append(classes[val])
 
    prot_ids = np.array(prot_ids)
    
    # Add corrupted triples nf3
    n_classes = len(classes)
    data['nf3_neg'] = []
    for c, r, d in data['nf3']:
        x = np.random.choice(prot_ids)
        while x == c:
            x = np.random.choice(prot_ids)
            
        y = np.random.choice(prot_ids)
        while y == d:
             y = np.random.choice(prot_ids)
        data['nf3_neg'].append((c, r,x))
        data['nf3_neg'].append((y, r, d))
        
    
    data['nf1'] = np.array(data['nf1'])
    data['nf3'] = np.array(data['nf3'])
    data['nf4'] = np.array(data['nf4'])
    data['top'] = np.array([classes['owl:Thing'],])
    data['nf3_neg'] = np.array(data['nf3_neg'])
                            
    for key, val in data.items():
        index = np.arange(len(data[key]))
        np.random.seed(seed=100)
        np.random.shuffle(index)
        data[key] = val[index]
    
    return data, classes, relations

class ELModel(tf.keras.Model):

    def __init__(self, nb_classes, nb_relations, embedding_size, batch_size, margin=0.01, reg_norm=1):
        super(ELModel, self).__init__()
        self.nb_classes = nb_classes
        self.nb_relations = nb_relations
        self.margin = margin
        self.reg_norm = reg_norm
        self.batch_size = batch_size
        self.inf = 5.0 # For top radius
        cls_weights = np.random.uniform(low=-1, high=1, size=(nb_classes, embedding_size + 1))
        cls_weights = cls_weights / np.linalg.norm(
            cls_weights, axis=1).reshape(-1, 1)
        rel_weights = np.random.uniform(low=-1, high=1, size=(nb_relations, embedding_size))
        rel_weights = rel_weights / np.linalg.norm(
            rel_weights, axis=1).reshape(-1, 1)
        self.cls_embeddings = tf.keras.layers.Embedding(
            nb_classes,
            embedding_size + 1,
            input_length=1,
            weights=[cls_weights,])
        self.rel_embeddings = tf.keras.layers.Embedding(
            nb_relations,
            embedding_size,
            input_length=1,
            weights=[rel_weights,])

            
    def call(self, input):
        """Run the model."""
        nf1,nf3, nf4,top, nf3_neg = input
        loss1 = self.nf1_loss(nf1)
        loss3 = self.nf3_loss(nf3)
        loss4 = self.nf4_loss(nf4)
        loss_top = self.top_loss(top)
        loss_nf3_neg = self.nf3_neg_loss(nf3_neg)
        loss = loss1 + loss3 + loss4 + loss_top + loss_nf3_neg
        return loss

    
    def reg(self, x):
        res = tf.abs(tf.norm(x, axis=1) - self.reg_norm)
        res = tf.reshape(res, [-1, 1])
        return res
        
    def nf1_loss(self, input):
        c = input[:, 0]
        r = input[:,1]
        d = input[:, 2]
        c = self.cls_embeddings(c)
        d = self.cls_embeddings(d)
        r = self.rel_embeddings(r)

        rc = tf.math.abs(c[:, -1])
        rd = tf.math.abs(d[:, -1])
        x1 = c[:, 0:-1]
        x2 = d[:, 0:-1]
        x3 = x1 + r
        euc = tf.norm(x3 - x2, axis=1)
        dst = tf.reshape(tf.nn.relu(euc + rc - rd - self.margin), [-1, 1])
        return dst + self.reg(x1) + self.reg(x2)
    
    def nf3_loss(self, input):
        # C subClassOf R some D
        c = input[:, 0]
        r = input[:, 1]
        d = input[:, 2]
        c = self.cls_embeddings(c)
        d = self.cls_embeddings(d)
        r = self.rel_embeddings(r)
        x1 = c[:, 0:-1]
        x2 = d[:, 0:-1]
        x3 = x1 + r

        rc = tf.math.abs(c[:, -1])
        rd = tf.math.abs(d[:, -1])
        euc = tf.norm(x3 - x2, axis=1)
        dst = tf.reshape(tf.nn.relu(euc + rc - rd - self.margin), [-1, 1])
        
        return dst + self.reg(x1) + self.reg(x2)

    def nf3_neg_loss(self, input):
        # C subClassOf R some D
        c = input[:, 0]
        r = input[:, 1]
        d = input[:, 2]
        c = self.cls_embeddings(c)
        d = self.cls_embeddings(d)
        r = self.rel_embeddings(r)
        x1 = c[:, 0:-1]
        x2 = d[:, 0:-1]
        
        x3 = x1 + r

        rc = tf.math.abs(c[:, -1])
        rd = tf.math.abs(d[:, -1])
        euc = tf.norm(x3 - x2, axis=1)
        dst = tf.reshape((-(euc - rc - rd) + self.margin), [-1, 1])
        
        return dst + self.reg(x1) + self.reg(x2)


    def nf4_loss(self, input):
        # R some C subClassOf D
        r = input[:, 0]
        c = input[:, 1]
        d = input[:, 2]
        c = self.cls_embeddings(c)
        d = self.cls_embeddings(d)
        r = self.rel_embeddings(r)
        rc = tf.reshape(tf.math.abs(c[:, -1]), [-1, 1])
        rd = tf.reshape(tf.math.abs(d[:, -1]), [-1, 1])
        sr = rc + rd
        x1 = c[:, 0:-1]
        x2 = d[:, 0:-1]
        
        # c - r should intersect with d
        x3 = x1 - r
        dst = tf.reshape(tf.norm(x3 - x2, axis=1), [-1, 1])
        dst_loss = tf.nn.relu(dst - sr - self.margin)
        return dst_loss + self.reg(x1) + self.reg(x2)
    

    #No disjoint loss

    def top_loss(self, input):
        d = input[:, 0]
        d = self.cls_embeddings(d)
        rd = tf.reshape(tf.math.abs(d[:, -1]), [-1, 1])
        return tf.math.abs(rd - self.inf)

class Generator(object):

    def __init__(self, data, batch_size=128, steps=100):
        self.data = data
        self.batch_size = batch_size
        self.steps = steps
        self.start = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        return self.next()

    def reset(self):
        self.start = 0

    def next(self):
        if self.start < self.steps:
            nf1_index = np.random.choice(
                self.data['nf1'].shape[0], self.batch_size)
            nf3_index = np.random.choice(
                self.data['nf3'].shape[0], self.batch_size)
            nf4_index = np.random.choice(
                self.data['nf4'].shape[0], self.batch_size)
            top_index = np.random.choice(
                self.data['top'].shape[0], self.batch_size)
            nf3_neg_index = np.random.choice(
                self.data['nf3_neg'].shape[0], self.batch_size)
            nf1 = self.data['nf1'][nf1_index]
            nf3 = self.data['nf3'][nf3_index]
            nf4 = self.data['nf4'][nf4_index]
            top = self.data['top'][top_index]
            nf3_neg = self.data['nf3_neg'][nf3_neg_index]
            labels = np.zeros((self.batch_size, 1), dtype=np.float32)
            self.start += 1
            return ([nf1, nf3, nf4,top, nf3_neg], labels)
        else:
            self.reset()

class MyModelCheckpoint(ModelCheckpoint):

    def __init__(self, *args, **kwargs):
        super(ModelCheckpoint, self).__init__()
        self.out_classes_file = kwargs.pop('out_classes_file')
        self.out_relations_file = kwargs.pop('out_relations_file')
        self.monitor = kwargs.pop('monitor')
        self.cls_list = kwargs.pop('cls_list')
        self.rel_list = kwargs.pop('rel_list')
        self.valid_data = kwargs.pop('valid_data')
        self.proteins = kwargs.pop('proteins')
        self.prot_index = list(self.proteins.values())
        self.prot_dict = {v: k for k, v in enumerate(self.prot_index)}
    
        self.best_rank = 100000
        
    def on_epoch_end(self, epoch, logs=None):
        # Save embeddings every 10 epochs
        current_loss = logs.get(self.monitor)
        if math.isnan(current_loss):
            print('NAN loss, stopping training')
            self.model.stop_training = True
            return
        el_model = self.model.layers[-1]
        cls_embeddings = el_model.cls_embeddings.get_weights()[0]
        rel_embeddings = el_model.rel_embeddings.get_weights()[0]

        prot_embeds = cls_embeddings[self.prot_index]
        prot_rs = prot_embeds[:, -1].reshape(-1, 1)
        prot_embeds = prot_embeds[:, :-1]
        mean_rank = 0
        n = len(self.valid_data)

        for c, r, d in self.valid_data:
            c, r, d = self.prot_dict[c], r, self.prot_dict[d]
            ec = prot_embeds[c, :]
            rc = prot_rs[c, :]
            er = rel_embeddings[r, :]
            ec += er

            dst = np.linalg.norm(prot_embeds - ec.reshape(1, -1), axis=1)
            dst = dst.reshape(-1, 1)
            res = np.maximum(0, dst - rc - prot_rs - el_model.margin)
            res = res.flatten()
            index = rankdata(res, method='average')
            rank = index[d]
            mean_rank += rank
            
        mean_rank /= n
        # fmean_rank /= n
        print(f'\n Validation {epoch + 1} {mean_rank}\n')
        if mean_rank < self.best_rank:
            self.best_rank = mean_rank
            print(f'\n Saving embeddings {epoch + 1} {mean_rank}\n')
            cls_file = self.out_classes_file
            rel_file = self.out_relations_file
            # Save embeddings of every thousand epochs
            # if (epoch + 1) % 1000 == 0:
            # cls_file = f'{cls_file}_{epoch + 1}.pkl'
            # rel_file = f'{rel_file}_{epoch + 1}.pkl'

            df = pd.DataFrame(
                {'classes': self.cls_list, 'embeddings': list(cls_embeddings)})
            df.to_pickle(cls_file)
        
            df = pd.DataFrame(
                {'relations': self.rel_list, 'embeddings': list(rel_embeddings)})
            df.to_pickle(rel_file)

def build_model(device,train_data,classes,relations,valid_data):
    proteins = {}#substitute for classes with subclass case
    for val in all_subcls:
        proteins[val] = classes[val]
    nb_classes = len(classes)
    nb_relations = len(relations)
    print("no. classes:",nb_classes)
    print("no. relations:",nb_relations)
    nb_train_data = 0
    for key, val in train_data.items():
        nb_train_data = max(len(val), nb_train_data)
    train_steps = int(math.ceil(nb_train_data / (1.0 * batch_size)))
    train_generator = Generator(train_data, batch_size, steps=train_steps)

    cls_dict = {v: k for k, v in classes.items()}
    rel_dict = {v: k for k, v in relations.items()}
    
    cls_list = []
    rel_list = []
    for i in range(nb_classes):
        cls_list.append(cls_dict[i])
    for i in range(nb_relations):
        rel_list.append(rel_dict[i])

    with tf.device('/' + device):
        nf1 = Input(shape=(3,), dtype=np.int32)
        nf3 = Input(shape=(3,), dtype=np.int32)
        nf4 = Input(shape=(3,), dtype=np.int32)

        top = Input(shape=(1,), dtype=np.int32)
        nf3_neg = Input(shape=(3,), dtype=np.int32)
        el_model = ELModel(nb_classes, nb_relations, embedding_size, batch_size, margin, reg_norm)
        out = el_model([nf1,nf3, nf4,top, nf3_neg])
        model = tf.keras.Model(inputs=[nf1, nf3, nf4,top, nf3_neg], outputs=out)
        optimizer = optimizers.Adam(lr=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
    # TOP Embedding
        top = classes.get('owl:Thing', None)
        checkpointer = MyModelCheckpoint(
            out_classes_file=out_classes_file,
            out_relations_file=out_relations_file,
            cls_list=cls_list,
            rel_list=rel_list,
            valid_data=valid_data,
            proteins=proteins,
            monitor='loss')
        
        logger = CSVLogger(loss_history_file)

        # Save initial embeddings
        cls_embeddings = el_model.cls_embeddings.get_weights()[0]
        rel_embeddings = el_model.rel_embeddings.get_weights()[0]
        
        cls_file = out_classes_file
        rel_file = out_relations_file

        df = pd.DataFrame(
            {'classes': cls_list, 'embeddings': list(cls_embeddings)})
        df.to_pickle(cls_file)

        df = pd.DataFrame(
            {'relations': rel_list, 'embeddings': list(rel_embeddings)})
        df.to_pickle(rel_file)
        
        model.fit_generator(
            train_generator,
            steps_per_epoch=train_steps,
            epochs=epochs,
            workers=12,
            callbacks=[logger, checkpointer])

gdata_file="NCI_data/nci_norm_mod.owl"
train_data_model, classes, relations = load_data(gdata_file)

valid_data_file="NCI_data/NCI_valid.txt"
valid_data_model = load_valid_data(valid_data_file, classes, relations)

margins = [-0.1,0,0.1]
embed_dims = [100,200,50]
batch_size =  256
device='gpu:0'
reg_norm=1
learning_rate=3e-4
epochs=1000
for d in embed_dims:
    embedding_size = d
    print("***************Embedding Dim:",embedding_size,'****************')
    for m in margins:
        margin = m
        print("**************Margin Loss:",margin,"***************")
        out_classes_file = f'NCI_results/EL/NCI_{embedding_size}_{margin}_{epochs}_cls.pkl'
        out_relations_file = f'NCI_results/EL/NCI_{embedding_size}_{margin}_{epochs}_rel.pkl'
        loss_history_file= f'NCI_results/EL/NCI_lossHis_{embedding_size}_{margin}_{epochs}.csv'
        build_model(device,train_data_model,classes,relations,valid_data_model)





