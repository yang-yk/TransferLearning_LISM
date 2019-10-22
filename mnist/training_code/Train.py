# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 15:40:45 2018

@author: yyk
"""

from __future__ import division

import tensorflow as tf
import Get_data
import OCN
import Option
import Save_data
import time
import os
import numpy as np
import Log

def train():   
    
    os.environ['CUDA_VISIBLE_DEVICES']=Option.GPU 
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
#    
    Log.Log(Option.log_path, 'w+', 1) # set log file
    print open('Option.py').read()
    #print open('OCN.py').read()
    
    
    batch_x=tf.placeholder(dtype=tf.float32,shape=[Option.batch_size,32,32,1],name='batch_x')
    batch_y=tf.placeholder(dtype=tf.float32,shape=[Option.batch_size,32,32,1],name='batch_y')
#    batch_pre=tf.placeholder(dtype=tf.float32,shape=[Option.batch_size,32,32,1],name='batch_pre')
    
    
    
    train_data_batch,train_label_batch=Get_data.read_and_decode(Option.train_tfrecords_path, batch_size=Option.batch_size,data_name='train_data',label_name='train_label')  
    valid_data_batch,valid_label_batch=Get_data.read_and_decode(Option.valid_tfrecords_path, batch_size=Option.batch_size,data_name='valid_data',label_name='valid_label') 
    test_data_batch,test_label_batch=Get_data.read_and_decode(Option.test_tfrecords_path, batch_size=Option.batch_size,data_name='test_data',label_name='test_label')
    
    predict_imgs=OCN.build(batch_x,Option.channel,is_training=True,fine_tuning=True,keep_prob=0.3)
    valid_imgs=OCN.build(batch_x,Option.channel,is_training=False,fine_tuning=False,keep_prob=1.0)
    test_imgs=OCN.build(batch_x,Option.channel,is_training=False,fine_tuning=False,keep_prob=1.0)
    
    train_loss,train_mse_loss,train_sharpness_loss,train_l2_loss,valid_loss,valid_mse_loss,valid_sharpness_loss,valid_l2_loss,test_loss,\
    test_mse_loss,test_sharpness_loss,test_l2_loss=Save_data.define_data()
    
    l2_loss_op=OCN.l2_loss(name='train_l2_loss_op')
    sharp_loss_op=OCN.sharpness(predict_imgs,batch_y,name='train_sharp_loss_op')
    mse_loss_op=OCN.mse_loss(predict_imgs,batch_y,name='train_mse_loss_op')    
    total_loss_op=tf.add(l2_loss_op+sharp_loss_op+mse_loss_op,0.0,name='train_total_loss_op')
    
    
    valid_l2_loss_op=l2_loss_op
    valid_sharp_loss_op=OCN.sharpness(valid_imgs,batch_y,name='valid_sharp_loss_op')
    valid_mse_loss_op=OCN.mse_loss(valid_imgs,batch_y,name='valid_mse_loss_op')    
    valid_total_loss_op=tf.add(valid_l2_loss_op+valid_sharp_loss_op+valid_mse_loss_op,0.0,name='valid_total_loss_op')
    
    test_l2_loss_op=l2_loss_op
    test_sharp_loss_op=OCN.sharpness(test_imgs,batch_y,name='valid_sharp_loss_op')
    test_mse_loss_op=OCN.mse_loss(test_imgs,batch_y,name='valid_mse_loss_op')    
    test_total_loss_op=tf.add(test_l2_loss_op+test_sharp_loss_op+test_mse_loss_op,0.0,name='valid_total_loss_op')   
    
    
    
    
    
    global_step = tf.Variable(0, name = 'global_step', trainable = False)
    global_step_op9 = tf.Variable(0, name = 'global_step_op9', trainable = False)
    
    
    train_op = OCN.train(total_loss_op, Option.learning_rate, Option.learning_rate_decay_steps, Option.learning_rate_decay_rate, global_step)    
    finetuning_op9=OCN.fine_tune9(total_loss_op,Option.learning_rate,Option.learning_rate_decay_steps,Option.learning_rate_decay_rate,global_step_op9,name='finetuning_op9')
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    
    
    saver = tf.train.Saver()
    
    if not os.path.exists(Option.result_path+'/log'):
        os.mkdir(Option.result_path+'/log')
    if not os.path.exists(Option.result_path+'/model'):
        os.mkdir(Option.result_path+'/model')
    
    sess=tf.Session(config=config)
    sess.run(init_op)
    sess.run(tf.local_variables_initializer())
      
    
    train_writer = tf.summary.FileWriter(Option.result_path+'/log', sess.graph)
    merged = tf.summary.merge_all()
    
    
    
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)   
    #sess.run(finetuning_op9) 
    
    start_time=time.time()
    
    valid_mse_loss_formodel=[]

    print("<<<<<<<<<<<<<<start training>>>>>>>>>>")
  

    try:
        while not coord.should_stop():
              step = tf.train.global_step(sess, global_step)     
              train_data_batch_,train_label_batch_=sess.run([train_data_batch,train_label_batch])  
              
              predict_imgs_,_,train_l2_loss_,train_sharp_loss_,train_mse_loss_,train_total_loss_,summary=sess.run([predict_imgs,train_op,l2_loss_op,sharp_loss_op,mse_loss_op,total_loss_op,merged],\
                feed_dict={batch_x:train_data_batch_,batch_y:train_label_batch_})
                         
              train_writer.add_summary(summary, step)

              train_loss,train_mse_loss,train_sharpness_loss,train_l2_loss=Save_data.train_list_append(train_loss,train_mse_loss,train_sharpness_loss,train_l2_loss,\
              train_total_loss_,train_l2_loss_,train_mse_loss_,train_sharp_loss_)
            
              epoch = step * Option.batch_size / Option.data_size
              batch_num=Option.data_size/Option.batch_size

              duration = time.time() - start_time
              print('[PROGRESS]\tEpoch %d, Step %d:  sharp_loss_=%3f , l2_loss=%.3f, mse_loss= %3f, loss = %.3f, (%.3f sec)' % (epoch, step%batch_num, train_sharp_loss_,\
              train_l2_loss_, train_mse_loss_, train_total_loss_, duration))
              
               
              if ( step%20==0 and step >0 ) or (step==int(Option.num_epochs*Option.data_size/Option.batch_size)):
                   
                   valid_data_batch_,valid_label_batch_=sess.run([valid_data_batch,valid_label_batch])
                   #valid_imgs_=sess.run(valid_imgs,feed_dict={batch_x:valid_data_batch_,batch_y:valid_label_batch_})
                   
                   
                   
                   valid_imgs_,valid_l2_loss_,valid_sharp_loss_,valid_mse_loss_,valid_total_loss_=sess.run([valid_imgs,valid_l2_loss_op,valid_sharp_loss_op,valid_mse_loss_op,valid_total_loss_op],\
                     feed_dict={batch_x:valid_data_batch_,batch_y:valid_label_batch_})
                     
                   valid_loss,valid_mse_loss,valid_sharpness_loss,valid_l2_loss=Save_data.valid_list_append(valid_loss,valid_mse_loss,valid_sharpness_loss,valid_l2_loss,\
                   valid_total_loss_,valid_l2_loss_,valid_mse_loss_,valid_sharp_loss_)
                   
                   print('Validiation Loss:  valid_sharp_loss_=%3f , valid_l2_loss=%.3f, valid_mse_loss= %3f, valid_loss = %.3f' % (valid_sharp_loss_,\
                   valid_l2_loss_, valid_mse_loss_, valid_total_loss_)) 
                   
                   
                   valid_mse_loss_formodel.append(valid_mse_loss_)
                   if min(valid_mse_loss_formodel)<Option.threshold and min(valid_mse_loss_formodel)==valid_mse_loss_:
                      print('[PROGRESS]\tSaving checkpoint')
                      checkpoint_path = os.path.join(Option.checkpoint_dir, 'unet.ckpt')
                      saver.save(sess, checkpoint_path, global_step = step)
                       
                   
                   
                   if step==int(Option.num_epochs*Option.data_size/Option.batch_size):
                   #if step==2: 
                       
                       
                      valid_imgs_=valid_imgs_.reshape(Option.batch_size,32,32)
                      valid_label_batch_=valid_label_batch_.reshape(Option.batch_size,32,32)
                      
                      Save_data.save_img(valid_imgs_,'valid_predict',number=Option.batch_size)
                      Save_data.save_img(valid_label_batch_,'valid_label',number=Option.batch_size)
                   
                                        
                       
                   
                   
                   
              if step==int(Option.num_epochs*Option.data_size/Option.batch_size):
              #if step==10:
                   print("End training...")    
                   
                   
                   
                   test_imgs_list=[]
                   label_img_list=[]
                   
                   testbatch_num=int(Option.test_size/Option.batch_size)
                   for i in range(int(testbatch_num)):
                       
                       test_data_batch_,test_label_batch_=sess.run([test_data_batch,test_label_batch])
                       #test_imgs_=sess.run(valid_imgs,feed_dict={batch_x:test_data_batch_,batch_y:test_label_batch_})
                       
                       if i%10==0:
                           print('test step %d' %(i))
                       test_imgs_,test_l2_loss_,test_sharp_loss_,test_mse_loss_,test_total_loss_=sess.run([test_imgs,test_l2_loss_op,test_sharp_loss_op,test_mse_loss_op,test_total_loss_op],\
                         feed_dict={batch_x:test_data_batch_,batch_y:test_label_batch_})
                       
                       test_imgs_list.append(test_imgs_) 
                       label_img_list.append(test_label_batch_)
                       
                       test_loss,test_mse_loss,test_sharpness_loss,test_l2_loss=Save_data.test_list_append(test_loss,test_mse_loss,\
                          test_sharpness_loss,test_l2_loss,test_total_loss_,test_l2_loss_,test_mse_loss_,test_sharp_loss_)                   
                       
                   
                   
                   Save_data.save_data(train_loss,train_mse_loss,train_sharpness_loss,train_l2_loss,valid_loss,valid_mse_loss,valid_sharpness_loss,valid_l2_loss,\
                            test_loss,test_sharpness_loss,test_l2_loss,test_mse_loss)
                   
                   label_img_list=np.array(label_img_list)
                   test_imgs_list=np.array(test_imgs_list)
                   
                   label_img_list=label_img_list.reshape(int(testbatch_num*Option.batch_size),32,32)
                   test_imgs_list=test_imgs_list.reshape(int(testbatch_num*Option.batch_size),32,32)
                   
                   
                   
                   Save_data.save_img(test_imgs_list,'test_predict',number=Option.image_number)
                   Save_data.save_img(label_img_list,'test_label',number=Option.image_number)
                   
                   
                   
                   
                   
                   np.save(Option.result_path+'/data/predict_imgs.npy',test_imgs_list)
                   np.save(Option.result_path+'/data/label_imgs.npy',label_img_list)
                   
                  
                   print("<<<<<<<<<<<<<<<<<Ending>>>>>>>>>>>>>>>>>>")                 
                   
                   
                   coord.request_stop()
                   print('[PROGRESS]\tSaving checkpoint')
                   checkpoint_path = os.path.join(Option.checkpoint_dir, 'unet.ckpt')
                   saver.save(sess, checkpoint_path, global_step = step)
                   print("<<<<<<<<<<<<<<<<<<ending>>>>>>>>>>>>>>>>>>>>>>>>")
                   exit(0)



#              if step % 1000==0 and step >0:
#                 print('[PROGRESS]\tSaving checkpoint')
#                 checkpoint_path = os.path.join(Option.checkpoint_dir, 'unet.ckpt')
#                 saver.save(sess, checkpoint_path, global_step = step)
    except  tf.errors.OutOfRangeError:
        print('[INFO    ]\tDone training for %d epochs, %d steps.' % (Option.num_epochs, step))
    finally:
         # When done, ask the threads to stop.
        coord.request_stop()
    coord.join(threads)
    sess.close()

if __name__=='__main__':
    train()
    
    
    
