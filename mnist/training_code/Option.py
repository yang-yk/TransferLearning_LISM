# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 15:01:53 2018

@author: yyk
"""

import numpy as np
import time
import os


batch_size=128

learning_rate=1e-3
learning_rate_decay_steps=100
learning_rate_decay_rate=0.99
data_size=50000
test_size=10000
num_epochs=100
channel=1
#sharpness_weight=0.0
sharpness_weight=0.008
#sharpness_weight=0
threshold=8

istraining=True
GPU='2'

L1_regularzation=True

feature_num=np.array([64,128,256,512,256,128,64,128,64])


train_tfrecords_path = '../tf_dataset/train_data.tfrecords'  
valid_tfrecords_path = '../tf_dataset/valid_data.tfrecords'  
test_tfrecords_path = '../tf_dataset/test_data.tfrecords'

exp_path='../training_experiment/'
Time=time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(time.time()))
if not os.path.isdir(exp_path):
   os.mkdir(exp_path)
os.mkdir(exp_path+Time,0755)
result_path=exp_path+Time

checkpoint_dir=result_path+'/model'
log_path=result_path+'/log.txt'

image_number=200
