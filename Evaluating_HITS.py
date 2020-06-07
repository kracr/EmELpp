#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import click as ck
import numpy as np
import pandas as pd
import logging
import math
import os
from collections import deque

from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
import matplotlib.pyplot as plt
from scipy.stats import rankdata

logging.basicConfig(level=logging.INFO)


# In[2]:


def load_eval_data(data_file):
    data = []
    rel = f'SubClassOf'
    with open(data_file, 'r') as f:
        for line in f:
            it = line.strip().split()
            id1 = it[0]
            id2 = it[1]
            data.append((id1, rel, id2))
    return data


# In[3]:


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


# In[4]:


def create_all_subcls(train_file,valid_file,test_file):
    total_sub_cls=[]
    train_sub_cls,train_samples = load_cls(train_file)
    valid_sub_cls,valid_samples = load_cls(valid_file)
    test_sub_cls,test_samples = load_cls(test_file)
    total_sub_cls = train_sub_cls + valid_sub_cls + test_sub_cls
    all_subcls = list(set(total_sub_cls))
    return all_subcls


# In[5]:


def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc

def compute_rank_roc(ranks, n_prots):
    auc_x = list(ranks.keys())
    auc_x.sort()
    auc_y = []
    tpr = 0
    sum_rank = sum(ranks.values())
    for x in auc_x:
        tpr += ranks[x]
        auc_y.append(tpr / sum_rank)
    auc_x.append(n_prots)
    auc_y.append(1)
    auc = np.trapz(auc_y, auc_x) / n_prots
    return auc

def compute_fmax(labels, preds):
    fmax = 0.0
    pmax = 0.0
    rmax = 0.0
    tmax = 0
    tpmax = 0
    fpmax = 0
    fnmax = 0
    for t in range(101):
        th = t / 100
        predictions = (preds >= th).astype(np.int32)
        tp = np.sum(labels & predictions)
        fp = np.sum(predictions) - tp
        fn = np.sum(labels) - tp
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        if p + r == 0:
            continue
        f = 2 * (p * r) / (p + r)
        if f > fmax:
            fmax = f
            pmax = p
            rmax = r
            tmax = t
            tpmax, fpmax, fnmax = tp, fp, fn
    return fmax, pmax, rmax, tmax, tpmax, fpmax, fnmax


# In[12]:


def compute_HITS(esize,mrg,cls_embeds_file,rel_embeds_file):
    embedding_size=esize
    reg_norm=1
    margin=mrg
    all_subcls = create_all_subcls(train_file,valid_file,test_file)
    cls_df = pd.read_pickle(cls_embeds_file)
    rel_df = pd.read_pickle(rel_embeds_file)
    nb_classes = len(cls_df)
    nb_relations = len(rel_df)
    embeds_list = cls_df['embeddings'].values
    classes = {v: k for k, v in enumerate(cls_df['classes'])}
    rembeds_list = rel_df['embeddings'].values
    relations = {v: k for k, v in enumerate(rel_df['relations'])}
    size = len(embeds_list[0])
    embeds = np.zeros((nb_classes, size), dtype=np.float32)
    for i, emb in enumerate(embeds_list):
        embeds[i, :] = emb
    
    subclasses= {}#substitute for classes with subclass case
    for val in all_subcls:
        subclasses[val] = classes[val]
    
    #radius of classes
    rs = np.abs(embeds[:, -1]).reshape(-1, 1) 
    #embeddings of classes
    embeds = embeds[:, :-1]
    subcls_index = list(subclasses.values())

    subcls_rs = rs[subcls_index, :]
    subcls_embeds = embeds[subcls_index, :]
    subcls_dict = {v: k for k, v in enumerate(subcls_index)}
    rsize = len(rembeds_list[0])
    rembeds = np.zeros((nb_relations, rsize), dtype=np.float32)
    for i, emb in enumerate(rembeds_list):
        rembeds[i, :] = emb
    
            
    top1 = 0
    top10 = 0
    top100 = 0
    mean_rank = 0
    ftop1 = 0
    ftop10 = 0
    ftop100 = 0
    fmean_rank = 0
    labels = {}
    preds = {}
    ranks = {}
    franks = {}
    eval_data = test_data
    rank_vals=[]
    crd_list=[]
    crd_index_list=[]
    with ck.progressbar(eval_data) as prog_data:
        for c, r, d in prog_data:
            if c not in classes or d not in classes:
                continue
            if classes[c] not in subcls_dict or classes[d] not in subcls_dict:
                continue
            c, r, d = subcls_dict[classes[c]], relations[r], subcls_dict[classes[d]]

            ec = subcls_embeds[c, :]
            rc = subcls_rs[c, :]
            er = rembeds[r, :]
            ec += er
        
            dst = np.linalg.norm(subcls_embeds - ec.reshape(1, -1), axis=1)
            dst = dst.reshape(-1, 1)
            
            res = np.maximum(0, dst - rc - subcls_rs - margin)
            res = res.flatten()

            index = rankdata(res, method='average')
            rank = index[d]
            crd_list.append((c,r,d))
            crd_index_list.append(index)
            if rank == 1:
                top1 += 1
            if rank <= 10:
                top10 += 1
            if rank <= 100:
                top100 += 1
            mean_rank += rank
            if rank not in ranks:
                ranks[rank] = 0
            ranks[rank] += 1
            rank_vals.append(rank)

        n = len(crd_list)
        top1 /= n
        top10 /= n
        top100 /= n
        mean_rank /= n
        ftop1 /= n
        ftop10 /= n
        ftop100 /= n
        fmean_rank /= n
    
    rank_auc = compute_rank_roc(ranks, len(subclasses))

    print(f'{embedding_size} {margin} {reg_norm} {top10:.2f} {top100:.2f} {mean_rank:.2f} {rank_auc:.2f}')

    return ranks,rank_vals,crd_list,crd_index_list


# In[13]:


def compute_rank_percentile(scores,x):
    scores.sort()
    per = np.percentile(scores,x)
    return per

import statistics
def compute_median_rank(rank_list):
    med = np.median(rank_list)
    return med    

def calculate_percentile_1000(scores):
    ranks_1000=[]
    for item in scores:
        if item < 1000:
            ranks_1000.append(item)
    n_1000 = len(ranks_1000)
    nt = len(scores)
    percentile = (n_1000/nt)*100
    return percentile


# In[14]:


def out_results(rks_vals):
    med_rank = compute_median_rank(rks_vals)
    print("Median Rank:",med_rank)
    per_rank_90 = compute_rank_percentile(rks_vals,90)
    print("90th percentile rank:",per_rank_90)
    percentile_below1000 = calculate_percentile_1000(rks_vals)
    print("Percentile for below 1000:",percentile_below1000)
    print("% Cases with rank greater than 1000:",(100 - percentile_below1000))


# In[15]:


def evaluation_HITS(data_dir,tag):
    margins = [-0.1,0,0.1]
    embed_dims = [100]
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
            out_classes_file = f'{tag}_{embedding_size}_{margin}_{epochs}_cls.pkl'
            out_relations_file = f'{tag}_{embedding_size}_{margin}_{epochs}_rel.pkl'
            cls_embeds_file=data_dir + out_classes_file
            rel_embeds_file=data_dir + out_relations_file
            ranks_bg,rank_vals_bg,crd_list_bg,crd_index_list_bg = compute_HITS(embedding_size,margin,cls_embeds_file,rel_embeds_file)
            print("Results----------->")
            out_results(rank_vals_bg)
               


#Parameters for GALEN
tag='GALEN'
GALENel_dir = "GALEN/GALEN_results/EL/"
GALEN_data = "GALEN/GALEN_data/"

train_file = GALEN_data + "GALLEN_train.txt"
valid_file = GALEN_data + "GALLEN_valid.txt"
test_file = GALEN_data + "GALLEN_test.txt"

train_data = load_eval_data(train_file)
valid_data = load_eval_data(valid_file)
test_data = load_eval_data(test_file)


# In[19]:


print("EL on test--------->")
evaluation_HITS(GALENel_dir,tag)


# In[20]:


print("EmEL on test--------->")
GALENEm_dir = "GALEN/GALEN_results/EmEL/"
evaluation_HITS(GALENEm_dir,tag)






