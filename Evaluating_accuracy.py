#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from sklearn.metrics import accuracy_score


# In[2]:


def load_cls(data_file):
    train_subs=list()
    counter=0
    with open(data_file,'r') as f:
        for line in f:
            counter+=1
            it = line.strip().split()
            cls1 = it[0]
            cls2 = it[1]
            train_subs.append(cls1)
            train_subs.append(cls2)
    train_cls = list(set(train_subs))
    return train_cls,counter


# In[3]:


def load_data(data_file):
    data = []
    rel = f'SubClassOf'
    with open(data_file, 'r') as f:
        for line in f:
            it = line.strip().split()
            id1 = it[0]
            id2 = it[1]
            data.append((id1, rel, id2))
    return data


# In[4]:


def circle(distSq, r1, r2): 
    flag=0
    if (distSq + r2 == r1): 
        flag=10
#         print("The smaller circle lies completely"
#             " inside the bigger circle with "
#             "touching each other "
#             "at a poof circumference. ") 
    elif (distSq + r2 < r1): 
        flag=11
#         print("The smaller circle lies completely"
#             " inside the bigger circle without"
#             " touching each other "
#             "at a poof circumference. ") 
    else: 
        flag=12
        if (distSq < r1 + r2):
            flag=13
            if (distSq + r1 > r2):
                flag=14
            
#         print("The smaller does not lies inside"
#             " the bigger circle completely.") 
    return flag


# In[5]:


def get_params(cls_embeds_file,rel_embeds_file):
    total_sub_cls=[]
    train_sub_cls,train_samples = load_cls(train_file)
    valid_sub_cls,valid_samples = load_cls(valid_file)
    test_sub_cls,test_samples = load_cls(test_file)
    total_sub_cls = train_sub_cls + valid_sub_cls + test_sub_cls
    all_subcls = list(set(total_sub_cls))
    
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
        
    return subcls_dict,subcls_embeds,subcls_rs,classes,relations 
    


# In[6]:


#Compute Accuracy
def compute_preds(data,subcls_dict,subcls_embeds,subcls_rs,classes,relations):
    cnt10=0
    cnt11=0
    cnt12=0
    cnt13=0
    cnt14=0
    acc_preds=[]
    acc_labels=[]
    less_cnt=0
    for c, r, d in data:
        c, r, d = subcls_dict[classes[c]], relations[r], subcls_dict[classes[d]]
        acc_labels.append(1)
        c_embeds = subcls_embeds[c,:]
        d_embeds = subcls_embeds[d,:]
        centers_dst = np.linalg.norm(d_embeds-c_embeds)
        c_radius = subcls_rs[c, :]
        d_radius = subcls_rs[d, :]

        if c_radius > d_radius:
#             print("1C_rad:",c_radius,"1D_rad:",d_radius)
            acc_preds.append(0)
            less_cnt+=1
            
        else:
            if centers_dst >= abs(d_radius-c_radius) or centers_dst <= (d_radius+c_radius):
                if centers_dst > (d_radius+c_radius):
#                     print("Dist:",centers_dst,"Small radius:",c_radius,"Bigger radius:",d_radius)
                    acc_preds.append(0)
                else:
                    flag = circle(centers_dst,d_radius,c_radius)
#                     print("flag--->",flag)
                    if flag==10:
                        cnt10+=1
                    if flag==11:
                        cnt11+=1
                    if flag==12:
                        cnt12+=1
                    if flag==13:
                        cnt13+=1
                    if flag==14:
                        cnt14+=1
                    
                    acc_preds.append(1)
            else:
                acc_preds.append(0)

#     print("10:",cnt10,"11:",cnt11,"12:",cnt12,"13:",cnt13,"14:",cnt14)
#     print("less cases:",less_cnt)
    return acc_preds,acc_labels


# In[7]:


def all_acc(cls_embeds_file,rel_embeds_file):
    subcls_dict,subcls_embeds,subcls_rs,classes,relations = get_params(cls_embeds_file,rel_embeds_file)
    tr_ap,tr_al = compute_preds(train_data,subcls_dict,subcls_embeds,subcls_rs,classes,relations)
    acc_tr = accuracy_score(tr_al,tr_ap)
    print("Training Accuracy:",acc_tr)
    va_ap,va_al = compute_preds(valid_data,subcls_dict,subcls_embeds,subcls_rs,classes,relations)
    acc_va = accuracy_score(va_al,va_ap)
    print("Validation Accuracy:",acc_va)
    te_ap,te_al = compute_preds(test_data,subcls_dict,subcls_embeds,subcls_rs,classes,relations)
    acc_te = accuracy_score(te_al,te_ap)
    print("Testing Accuracy:",acc_te)
    i_ap,i_al = compute_preds(itest_data,subcls_dict,subcls_embeds,subcls_rs,classes,relations)
    acc_i = accuracy_score(i_al,i_ap)
    print("Inferences Accuracy:",acc_i)


# In[8]:


def evaluation_acc(data_dir,tag):
    margins = [-0.1,0,0.1]
    embed_dims = [100,50]
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
            all_acc(cls_embeds_file,rel_embeds_file)
           


# # Results on GALEN

# In[16]:


#Evaluating for the best models
#Parameters
tag='GALEN'
AEL_dir = f'{tag}/EmEL/'
margin = 0
embedding_size = 50
batch_size =  256
device='gpu:0'
reg_norm=1
learning_rate=3e-4



cls_embeds_file = AEL_dir + f'{tag}_{embedding_size}_{margin}_1000_cls.pkl'
rel_embeds_file = AEL_dir + f'{tag}_{embedding_size}_{margin}_1000_rel.pkl'
train_file = f'{tag}/{tag}_train.txt'
valid_file = f'{tag}/{tag}_valid.txt'
test_file = f'{tag}/{tag}_test.txt'
inference_file = f'{tag}/{tag}_inferences.txt'

train_data = load_data(train_file)
valid_data = load_data(valid_file)
test_data = load_data(test_file)
itest_data = load_data(inference_file)


# In[17]:


print("EmEL GALEN accuracy results--------->")
all_acc(cls_embeds_file,rel_embeds_file)


# In[18]:


AEL_dir = f'{tag}/EL/'
margin = 0
embedding_size = 50
batch_size =  256
device='gpu:0'
reg_norm=1
learning_rate=3e-4
cls_embeds_file = AEL_dir + f'{tag}_{embedding_size}_{margin}_1000_cls.pkl'
rel_embeds_file = AEL_dir + f'{tag}_{embedding_size}_{margin}_1000_rel.pkl'
print("EL GALEN accuracy results--------->")
all_acc(cls_embeds_file,rel_embeds_file)


# # Results on GO

# In[21]:


#Evaluating for directions
#Parameters 
tag='GO'
AEL_dir = f'{tag}/EmEL/'
margin = -0.1
embedding_size = 100
batch_size =  256
device='gpu:0'
reg_norm=1
learning_rate=3e-4



cls_embeds_file = AEL_dir + f'{tag}_{embedding_size}_{margin}_1000_cls.pkl'
rel_embeds_file = AEL_dir + f'{tag}_{embedding_size}_{margin}_1000_rel.pkl'
train_file = f'{tag}/{tag}_train.txt'
valid_file = f'{tag}/{tag}_valid.txt'
test_file = f'{tag}/{tag}_test.txt'
inference_file = f'{tag}/{tag}_inferences.txt'

train_data = load_data(train_file)
valid_data = load_data(valid_file)
test_data = load_data(test_file)
itest_data = load_data(inference_file)


# In[22]:


print("EmEL GO accuracy results--------->")
all_acc(cls_embeds_file,rel_embeds_file)


# In[23]:


AEL_dir = f'{tag}/EL/'
margin = -0.1
embedding_size = 100
batch_size =  256
device='gpu:0'
reg_norm=1
learning_rate=3e-4
cls_embeds_file = AEL_dir + f'{tag}_{embedding_size}_{margin}_1000_cls.pkl'
rel_embeds_file = AEL_dir + f'{tag}_{embedding_size}_{margin}_1000_rel.pkl'
print("EL GO accuracy results--------->")
all_acc(cls_embeds_file,rel_embeds_file)


# # Results on ANATOMY

# In[28]:


#Evaluating for directions
#Parameters 
tag='ANATOMY'
AEL_dir = f'{tag}/EmEL/'
margin = -0.1
embedding_size = 200
batch_size =  256
device='gpu:0'
reg_norm=1
learning_rate=3e-4



cls_embeds_file = AEL_dir + f'{tag}_{embedding_size}_{margin}_1000_cls.pkl'
rel_embeds_file = AEL_dir + f'{tag}_{embedding_size}_{margin}_1000_rel.pkl'
train_file = f'{tag}/{tag}_train.txt'
valid_file = f'{tag}/{tag}_valid.txt'
test_file = f'{tag}/{tag}_test.txt'
inference_file = f'{tag}/{tag}_inferences.txt'

train_data = load_data(train_file)
valid_data = load_data(valid_file)
test_data = load_data(test_file)
itest_data = load_data(inference_file)


# In[29]:


print("EmEL ANATOMY accuracy results--------->")
all_acc(cls_embeds_file,rel_embeds_file)


# In[30]:


AEL_dir = f'{tag}/EL/'
margin = -0.1
embedding_size = 200
batch_size =  256
device='gpu:0'
reg_norm=1
learning_rate=3e-4
cls_embeds_file = AEL_dir + f'{tag}_{embedding_size}_{margin}_1000_cls.pkl'
rel_embeds_file = AEL_dir + f'{tag}_{embedding_size}_{margin}_1000_rel.pkl'
print("EL ANATOMY accuracy results--------->")
all_acc(cls_embeds_file,rel_embeds_file)


# # Results on SNOMED

# In[33]:


tag='SNOMED'
train_file = f'{tag}/{tag}_train.txt'
# valid_file = f'{tag}/{tag}_valid.txt'
i_file = f'{tag}/itest_snomed.txt'
train_data = load_data(train_file)
idata = load_data(i_file)

A = set(train_data)
B = set(idata)
i_set = B - A
itest_data = list(i_set)
with open('crd_values_snomed.txt', 'w') as f:
    for item in itest_data:
        istr = item[0] +" " +item[2]
        f.write(istr)
        f.write("\n")


# In[34]:


#Evaluating for directions
#Parameters 
tag='SNOMED'
AEL_dir = f'{tag}/EmEL/'
margin = -0.1
embedding_size = 100
batch_size =  256
device='gpu:0'
reg_norm=1
learning_rate=3e-4



cls_embeds_file = AEL_dir + f'{tag}_{embedding_size}_{margin}_1000_cls.pkl'
rel_embeds_file = AEL_dir + f'{tag}_{embedding_size}_{margin}_1000_rel.pkl'
train_file = f'{tag}/{tag}_train.txt'
valid_file = f'{tag}/{tag}_valid.txt'
test_file = f'{tag}/{tag}_test.txt'
inference_file = f'{tag}/{tag}_inferences.txt'

train_data = load_data(train_file)
valid_data = load_data(valid_file)
test_data = load_data(test_file)
itest_data = load_data(inference_file)


# In[35]:


print("EmEL SNOMED accuracy results--------->")
all_acc(cls_embeds_file,rel_embeds_file)


# In[36]:


AEL_dir = f'{tag}/EL/'
margin = -0.1
embedding_size = 100
batch_size =  256
device='gpu:0'
reg_norm=1
learning_rate=3e-4
cls_embeds_file = AEL_dir + f'{tag}_{embedding_size}_{margin}_1000_cls.pkl'
rel_embeds_file = AEL_dir + f'{tag}_{embedding_size}_{margin}_1000_rel.pkl'
print("EL SNOMED accuracy results--------->")
all_acc(cls_embeds_file,rel_embeds_file)






