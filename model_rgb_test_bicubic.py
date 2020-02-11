from utils import (
  read_data,  
  imsave,
  merge
)
from skimage import io,data,color
import time
import os
import matplotlib.pyplot as plt
from ops import * 
import numpy as np
import tensorflow as tf
import scipy.io as sio
import skimage.filters.rank as sfr
class SRCNN(object):

  def __init__(self, 
               sess, 
               image_size=128,
               label_size=128, 
               batch_size=32,
               c_dim=1, 
               i = 0,
               h0=None,
               w0=None,
               checkpoint_dir=None, 
               sample_dir=None,
               test_dir=None,
               test_depth=None,
               test_label=None
               ):

    self.sess = sess
    self.is_grayscale = (c_dim == 1)

    self.image_size = image_size
    self.label_size = label_size
    self.batch_size = batch_size
    self.h = h0
    self.w = w0
    self.c_dim = c_dim
    self.i=i
    self.test_dir=test_dir
    self.test_depth=test_depth
    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    self.test_label = test_label  
    self.build_model()

  def build_model(self):
    self.images = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, 1], name='images')
    self.labels = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.label_size, self.label_size, 1], name='labels')
    self.I_add = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.label_size, self.label_size, self.c_dim], name='I_add')



    self.depth_test = tf.compat.v1.placeholder(tf.float32, [1, self.h, self.w, 1], name='images_test')
    self.I_add_test = tf.compat.v1.placeholder(tf.float32, [1, self.h, self.w, self.c_dim], name='I_add_test')
    if self.i < 1 :
      self.pred = self.model()
    self.pred_test = self.model_test()

    if self.test_dir is None:
    
      self.loss_mse = tf.reduce_mean(tf.square(self.labels - self.pred))
      self.loss = self.loss_mse

    self.saver = tf.compat.v1.train.Saver(max_to_keep=0)

  def train(self, config):
    if config.is_train:
      A=1;
    else:

      I_add_input_test = imread(self.test_dir, is_grayscale=True)/255   
      
      image_path = os.path.join(config.sample_dir, str(self.i)+"_rgb.png" )
      imsave(I_add_input_test, image_path)

      I_add_input_test=I_add_input_test.reshape([1,self.h,self.w,self.c_dim])
      depth_label = sio.loadmat(self.test_label)['I_up']
      depth_down = sio.loadmat(self.test_depth)['I_down'].astype(np.float)
      
      max_label=np.max(depth_label)
      min_label=np.min(depth_label)
      max_down=np.max(depth_down)
      min_down=np.min(depth_down)

      image_path = os.path.join(config.sample_dir, str(self.i)+"_down.png" )
      imsave(depth_down, image_path)
      depth_down=depth_down.reshape([1, self.h, self.w, 1])


    if config.is_train:     
      data_dir = config.data_path
      depth_input_down_list=glob.glob(os.path.join(data_dir,'*_patch_depth_down.mat'))
      depth_label_list     =glob.glob(os.path.join(data_dir,'*_patch_depth_label.mat'))
      rgb_input_list       =glob.glob(os.path.join(data_dir,'*_patch_I_add.mat'))

      

      seed=545
      np.random.seed(seed)
      np.random.shuffle(depth_input_down_list)
      np.random.seed(seed)
      np.random.shuffle(depth_label_list)
      np.random.seed(seed)
      np.random.shuffle(rgb_input_list)

      depth_input_down_list_test=glob.glob(os.path.join(data_dir,'patch_depth_down_test.mat'))
      depth_label_list_test     =glob.glob(os.path.join(data_dir,'patch_depth_label_test.mat'))
      rgb_input_list_test       =glob.glob(os.path.join(data_dir,'patch_I_add_test.mat'))
      
      self.train_op = tf.train.AdamOptimizer(config.learning_rate,0.9).minimize(self.loss)


    tf.initialize_all_variables().run()
    
    counter = 0
    start_time = time.time()
    if self.load(self.checkpoint_dir):
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    if config.is_train:
      print("Training...")
      loss = []
      for ep in range(config.epoch):
        #print('ep'+ str(ep))
        batch_idxs=len(depth_input_down_list)
        #print(batch_idxs)
        for idx in range(0,batch_idxs):
          time_image = time.time()
          #print('idx'+str(idx))
          batch_depth_down=get_image_batch_new(depth_input_down_list[idx])
          batch_depth_labels=get_image_batch_new(depth_label_list[idx])
          batch_I_add = get_image_batch_new(rgb_input_list[idx])/255
          counter += 1
          _, err = self.sess.run([self.train_op, self.loss], feed_dict={self.images: batch_depth_down, self.labels: batch_depth_labels,self.I_add:batch_I_add})
          #print("test-------Epoch: [%2d], step: [%2d], image: [%2d], time: [%4.4f], loss: [%.8f]" \
          #     % ((ep+1), counter,idx, time.time()-time_image, err))
          
          if counter % 1000 == 0:
            print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
              % ((ep+1), counter, time.time()-start_time, err))

          if idx == batch_idxs-1:
          # if counter % 10 == 0:
            batch_test_idxs = len(depth_input_down_list_test) // config.batch_size
            err_test =  np.ones(batch_test_idxs)
            for idx_test in range(0,batch_test_idxs):
              batch_depth_down = get_image_batch(depth_input_down_list_test, idx_test*config.batch_size , (idx_test+1)*config.batch_size)
              batch_depth_labels = get_image_batch(depth_label_list_test, idx_test*config.batch_size , (idx_test+1)*config.batch_size)
              batch_I_add = get_image_batch(rgb_input_list_test, idx_test*config.batch_size , (idx_test+1)*config.batch_size)/255
              err_test[idx_test] = self.sess.run(self.loss, feed_dict={self.images: batch_depth_down, self.labels: batch_depth_labels,self.I_add:batch_I_add})    

            loss.append(np.mean(err_test))
            print("test-------Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
               % ((ep+1), counter, time.time()-start_time, err))
               
            print(loss)
            self.save(config.checkpoint_dir, counter)


    else:
      print("Testing...")
      tm = time.time()
      
      #if config.save_parameters:
      result, residual_map,conv1_f, conv3_f, conv5_f, conv7_f, w1_f, w3_f, w5_f, w7_f, bias1_f, bias3_f, bias5_f, bias7_f, \
          pool1_f, pool2_f, pool3_f, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, w1, w2, w3, w4, w5, w6, w7,w8, w9, w10,\
          bias1, bias2, bias3, bias4, bias5, bias6, bias7, bias8, bias9, bias10, pool1, pool2, pool3, pool4, pool1_input, pool2_input,pool3_input,\
          pool4_input, conv_input1, conv_input2, conv_input3, conv_input4, w_input1, w_input2,w_input3,w_input4,deconv2,deconv3,deconv4,deconv5,\
          conv13, conv14, conv15, conv16,conv17,conv18,conv19,conv20, w13, w14, w15, w16,w17,w18,w19,w20 = self.sess.run(self.pred_test, feed_dict= {self.depth_test:depth_down ,self.I_add_test:I_add_input_test})
     
      print('Rec time ', time.time() - tm)
      
      result = np.minimum(np.maximum(result.squeeze(),0),1)

      result = ((result)*(max_label-min_label)+min_label).astype(np.uint8)
      print('TYPES !!!!')
      print(depth_label.dtype)
      rmse_value = rmse(depth_label,result) 
      
      print("rmse: [%f]" % rmse_value)
      image_path = config.results_path     
      image_path = os.path.join(image_path, str(self.i)+"_sr.png" )
      imsave(result, image_path)
      
      #initial rmse
      init_image = np.minimum(np.maximum(depth_down.squeeze(),0),1)
      init_image = ((init_image)*(max_label-min_label)+min_label).astype(np.uint8)
      init_rmse = rmse(depth_label, init_image)   
      print("initial rmse: [%f]" % init_rmse)
      
      if config.save_parameters:
         # create dictionary 
         dictionary = {'encoder_conv':{'conv1':conv1, 'conv2':conv2, 'conv3':conv3,'conv4':conv4, 'conv5':conv5, 'conv6':conv6,'conv7' : conv7, 'conv8':conv8, 'conv9':conv9, 'conv10':conv10},\
           'encoder_w':{'w1':w1, 'w2':w2, 'w3':w3, 'w4':w4, 'w5':w5, 'w6':w6, 'w7':w7,'w8':w8, 'w9':w9, 'w10':w10}, \
           'decoder_conv':{'deconv2':deconv2,'deconv3':deconv3, 'deconv4':deconv4, 'deconv5':deconv5, 'conv13':conv13,\
                      'conv14' : conv14, 'conv15':conv15, 'conv16':conv16, 'conv17':conv17,'conv18' : conv18, 'conv19':conv19, 'conv20':conv20}, \
           'decoder_w':{'w13':w13, 'w14':w14, 'w15':w15, 'w16':w16,'w17':w17,'w18':w18,'w19':w19 ,'w20':w20}, \
           'I_branch_F_conv':{'conv1_f':conv1_f, 'conv3_f':conv3_f,'conv5_f':conv5_f,'conv7_f' : conv7_f}, \
           'I_branch_F_w_f':{'w1_f':w1_f,'w3_f':w3_f, 'w5_f':w5_f, 'w7_f':w7_f}, \
           'Input_Pyramid_conv':{'conv_input1':conv_input1, 'conv_input2':conv_input2, 'conv_input3':conv_input3, 'conv_input4':conv_input4},\
           'Input_Pyramid_w':{'w_input1':w_input1, 'w_input2':w_input2, 'w_input3':w_input3, 'w_input4':w_input4}, \
           'rmse':{'init_rmse' : init_rmse, 'rmse' : rmse_value}, \
           'inputs':{'I_init': I_add_input_test, 'D_down':depth_down, 'D_label':depth_label}, \
           'results':{'result' : result, 'residual_map':residual_map}, \
           'Rec time ': time.time() - tm}
      
         sio.savemat(os.path.join(config.results_path, 'parameters.mat'), dictionary)
  


      return(rmse_value)

    
  def model(self):
    with tf.compat.v1.variable_scope("I_branch_F") as scope1:
      conv1_f,w1_f,bias1_f = conv2d(self.I_add, 1,64,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_1")
      conv1_f = tf.nn.relu(conv1_f)
      pool1_f=max_pool_2x2(conv1_f)
      conv3_f,w3_f, bias3_f = conv2d(pool1_f, 64,128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_3")
      conv3_f = tf.nn.relu(conv3_f)
      pool2_f=max_pool_2x2(conv3_f)
      conv5_f,w5_f, bias5_f = conv2d(pool2_f, 128,256, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_5")
      conv5_f = tf.nn.relu(conv5_f)
      pool3_f=max_pool_2x2(conv5_f)
      conv7_f, w7_f, bias7_f = conv2d(pool3_f,  1024,512, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_7")
      conv7_f = tf.nn.relu(conv7_f)
      
      
    with tf.compat.v1.variable_scope("main_branch") as scope3:
      conv1, w1, bias1 = conv2d(self.images, 1,64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_1")
      conv1 = tf.nn.relu(conv1)
      conv2, w2, bias2 = conv2d(conv1, 64,64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_2")
      conv2 = tf.nn.relu(conv2)
      pool1=max_pool_2x2(conv2)
      
      pool1_input=max_pool_2x2(self.images)
      conv_input1, w_input1, bias_input1 = conv2d(pool1_input, 1,64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_input1")
      conv_input1 = tf.nn.relu(conv_input1)
      concate_input1=tf.concat(axis = 3, values = [pool1,conv_input1])  

      conv3, w3, bias3 = conv2d(concate_input1, 128,128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_3")
      conv3 = tf.nn.relu(conv3)
      
      conv4, w4, bias4 = conv2d(conv3, 128,128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_4")
      conv4 = tf.nn.relu(conv4)
      pool2=max_pool_2x2(conv4)

      pool2_input=max_pool_2x2(pool1_input)
      conv_input2, w_input2, bias_input2 = conv2d(pool2_input, 1,128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_input2")
      conv_input2 = tf.nn.relu(conv_input2)
      concate_input2=tf.concat(axis = 3, values = [pool2,conv_input2])  

      conv5, w5, bias5 = conv2d(concate_input2, 256,256, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_5")
      conv5 = tf.nn.relu(conv5)
      conv6, w6, bias6 = conv2d(conv5, 256,256, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_6")
      conv6 = tf.nn.relu(conv6)
      pool3 = max_pool_2x2(conv6)

      pool3_input=max_pool_2x2(pool2_input)
      conv_input3, w_input3, bias_input3 = conv2d(pool3_input, 1,256, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_input3")
      conv_input3 = tf.nn.relu(conv_input3)
      concate_input3=tf.concat(axis = 3, values = [pool3,conv_input3])        

      conv7, w7, bias7 = conv2d(concate_input3, 512,512, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_7")
      conv7 = tf.nn.relu(conv7)
      conv8, w8, bias8 = conv2d(conv7, 512,512, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_8")
      conv8 = tf.nn.relu(conv8)
      pool4=max_pool_2x2(conv8)
     
      pool4_input=max_pool_2x2(pool3_input)
      conv_input4, w_input4, bias_input4 = conv2d(pool4_input, 1,512, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_input4")
      conv_input4 = tf.nn.relu(conv_input4)
      concate_input4=tf.concat(axis = 3, values = [pool4,conv_input4])        

      conv9, w9, bias9 = conv2d(concate_input4, 1024,1024, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_9")
      conv9 = tf.nn.relu(conv9)
      conv10, w10, bias10 = conv2d(conv9, 1024,1024, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_10")
      conv10 = tf.nn.relu(conv10)
      

      deconv2 = tf.nn.relu(deconv2d(conv10, conv8.get_shape().as_list(), k_h=3, k_w=3, d_h=2, d_w=2,name="deconv2d_2")) 
      conb2 = tf.concat(axis = 3, values = [deconv2,conv8,conv7_f])  
      conv13, w13, bias13 = conv2d(conb2, 3072,512, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_13")
      conv13 = tf.nn.relu(conv13)
      conv14, w14, bias14 = conv2d(conv13, 512,512, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_14")
      conv14 = tf.nn.relu(conv14)
      deconv3 = tf.nn.relu(deconv2d(conv14, conv6.get_shape().as_list(), k_h=3, k_w=3, d_h=2, d_w=2,name="deconv2d_3"))                          
      conb3 = tf.concat(axis = 3, values = [deconv3,conv6,conv5_f])
      conv15, w15, bias15 = conv2d(conb3, 768,256, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_15")
      conv15 = tf.nn.relu(conv15)      
      conv16,w16, bias16 = conv2d(conv15, 256,256, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_16")
      conv16 = tf.nn.relu(conv16)
      deconv4 = tf.nn.relu(deconv2d(conv16, conv4.get_shape().as_list(), k_h=3, k_w=3, d_h=2, d_w=2,name="deconv2d_4"))    
      conb4 = tf.concat(axis = 3, values = [deconv4,conv4,conv3_f])
      conv17, w17, bias17 = conv2d(conb4, 384,128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_17")
      conv17 = tf.nn.relu(conv17)      
      conv18, w18, bias18 = conv2d(conv17, 128,128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_18")
      conv18 = tf.nn.relu(conv18)
      deconv5 = tf.nn.relu(deconv2d(conv18, conv2.get_shape().as_list(), k_h=3, k_w=3, d_h=2, d_w=2,name="deconv2d_5"))
      conb5 = tf.concat(axis = 3, values = [deconv5,conv2,conv1_f])
      conv19, w19, bias19 = conv2d(conb5, 192,64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_19")
      conv19 = tf.nn.relu(conv19)      
      conv20, w20, bias20 = conv2d(conv19, 64,64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_20")
      conv20 = tf.nn.relu(conv20) 
      residual_map, w_output, bias_output = conv2d(conv20, 64,1, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_21")
      output = tf.add(residual_map,self.images) 
    return output
    #, residual_map,conv1_f, conv3_f, conv5_f, conv7_f, w1_f, w3_f, w5_f, w7_f, bias1_f, bias3_f, bias5_f, bias7_f, \
#pool1_f, pool2_f, pool3_f, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, w1, w2, w3, w4, w5, w6, w7, w8,w9, w10,\
#bias1, bias2, bias3, bias4, bias5, bias6, bias7, bias8, bias9, bias10, pool1, pool2, pool3, pool4, pool1_input, pool2_input,pool3_input,\
#pool4_input, conv_input1, conv_input2, conv_input3, conv_input4, w_input1, w_input2,w_input3,w_input4,deconv2,deconv3,deconv4,deconv5,\
#conv13, conv14, conv15, conv16,conv17,conv18,conv19,conv20, w13, w14,w15,w16,w17,w18,w19,w20 


  def model_test(self):
    with tf.compat.v1.variable_scope("I_branch_F", reuse = True):
      conv1_f,w1_f,bias1_f = conv2d(self.I_add_test, 1,64,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_1")
      conv1_f = tf.nn.relu(conv1_f)
      pool1_f=max_pool_2x2(conv1_f)
      conv3_f,w3_f, bias3_f = conv2d(pool1_f, 64,128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_3")
      conv3_f = tf.nn.relu(conv3_f)
      pool2_f=max_pool_2x2(conv3_f)
      conv5_f,w5_f, bias5_f = conv2d(pool2_f, 128,256, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_5")
      conv5_f = tf.nn.relu(conv5_f)
      pool3_f=max_pool_2x2(conv5_f)
      conv7_f,w7_f, bias7_f = conv2d(pool3_f, 1024,512, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_7")
      conv7_f = tf.nn.relu(conv7_f)

    with tf.compat.v1.variable_scope("main_branch", reuse = True):
      conv1, w1, bias1 = conv2d(self.depth_test, 1,64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_1")
      conv1 = tf.nn.relu(conv1)
      conv2, w2, bias2 = conv2d(conv1, 64,64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_2")
      conv2 = tf.nn.relu(conv2)
      pool1=max_pool_2x2(conv2)
      
      pool1_input=max_pool_2x2(self.depth_test)
      conv_input1, w_input1, bias_input1 = conv2d(pool1_input, 1,64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_input1")
      conv_input1 = tf.nn.relu(conv_input1)
      concate_input1=tf.concat(axis = 3, values = [pool1,conv_input1])  

      conv3, w3, bias3 = conv2d(concate_input1, 128,128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_3")
      conv3 = tf.nn.relu(conv3)
      conv4, w4, bias4 = conv2d(conv3, 128,128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_4")
      conv4 = tf.nn.relu(conv4)
      pool2=max_pool_2x2(conv4)

      pool2_input=max_pool_2x2(pool1_input)
      conv_input2, w_input2, bias_input2 = conv2d(pool2_input, 1,128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_input2")
      conv_input2 = tf.nn.relu(conv_input2)
      concate_input2=tf.concat(axis = 3, values = [pool2,conv_input2])  

      conv5, w5, bias5 = conv2d(concate_input2, 256,256, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_5")
      conv5 = tf.nn.relu(conv5)
      conv6, w6, bias6  = conv2d(conv5, 256,256, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_6")
      conv6 = tf.nn.relu(conv6)
      pool3=max_pool_2x2(conv6)

      pool3_input=max_pool_2x2(pool2_input)
      conv_input3, w_input3, bias_input3 = conv2d(pool3_input, 1,256, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_input3")
      conv_input3 = tf.nn.relu(conv_input3)
      concate_input3=tf.concat(axis = 3, values = [pool3,conv_input3])        

      conv7, w7, bias7 = conv2d(concate_input3, 512,512, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_7")
      conv7 = tf.nn.relu(conv7)
      conv8, w8, bias8 = conv2d(conv7, 512,512, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_8")
      conv8 = tf.nn.relu(conv8)
      pool4=max_pool_2x2(conv8)
     
      pool4_input=max_pool_2x2(pool3_input)
      conv_input4, w_input4, bias_input4 = conv2d(pool4_input, 1,512, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_input4")
      conv_input4 = tf.nn.relu(conv_input4)
      concate_input4=tf.concat(axis = 3, values = [pool4,conv_input4])        

      conv9, w9, bias9 = conv2d(concate_input4, 1024,1024, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_9")
      conv9 = tf.nn.relu(conv9)
      conv10, w10, bias10 = conv2d(conv9, 1024,1024, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_10")
      conv10 = tf.nn.relu(conv10)
      

      deconv2 = tf.nn.relu(deconv2d(conv10, conv8.get_shape().as_list(), k_h=3, k_w=3, d_h=2, d_w=2,name="deconv2d_2")) 
      conb2 = tf.concat(axis = 3, values = [deconv2,conv8,conv7_f])  
      conv13, w13, bias13 = conv2d(conb2, 3072,512, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_13")
      conv13 = tf.nn.relu(conv13)
      conv14, w14, bias14 = conv2d(conv13, 512,512, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_14")
      conv14 = tf.nn.relu(conv14)
      deconv3 = tf.nn.relu(deconv2d(conv14, conv6.get_shape().as_list(), k_h=3, k_w=3, d_h=2, d_w=2,name="deconv2d_3"))                          
      conb3 = tf.concat(axis = 3, values = [deconv3,conv6,conv5_f])
      conv15, w15, bias15 = conv2d(conb3, 768,256, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_15")
      conv15 = tf.nn.relu(conv15)      
      conv16, w16, bias16 = conv2d(conv15, 256,256, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_16")
      conv16 = tf.nn.relu(conv16)
      deconv4 = tf.nn.relu(deconv2d(conv16, conv4.get_shape().as_list(), k_h=3, k_w=3, d_h=2, d_w=2,name="deconv2d_4"))    
      conb4 = tf.concat(axis = 3, values = [deconv4,conv4,conv3_f])
      conv17, w17, bias17 = conv2d(conb4, 384,128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_17")
      conv17 = tf.nn.relu(conv17)      
      conv18, w18, bias18 = conv2d(conv17, 128,128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_18")
      conv18 = tf.nn.relu(conv18) 
      deconv5 = tf.nn.relu(deconv2d(conv18, conv2.get_shape().as_list(), k_h=3, k_w=3, d_h=2, d_w=2,name="deconv2d_5"))
      conb5 = tf.concat(axis = 3, values = [deconv5,conv2,conv1_f])
      conv19, w19, bias19 = conv2d(conb5, 192,64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_19")
      conv19 = tf.nn.relu(conv19)      
      conv20, w20, bias20 = conv2d(conv19, 64,64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_20")
      conv20 = tf.nn.relu(conv20) 
      residual_map, w_output, bias_output = conv2d(conv20, 64,1, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_21")
      output = tf.add(residual_map,self.depth_test) 
      
    return output,residual_map, conv1_f, conv3_f, conv5_f, conv7_f, w1_f, w3_f, w5_f, w7_f, bias1_f, bias3_f, bias5_f, bias7_f, \
pool1_f, pool2_f, pool3_f, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10,\
bias1, bias2, bias3, bias4, bias5, bias6, bias7, bias8, bias9, bias10, pool1, pool2, pool3, pool4, pool1_input, pool2_input,pool3_input,\
pool4_input, conv_input1, conv_input2, conv_input3, conv_input4, w_input1, w_input2,w_input3,w_input4,deconv2,deconv3,deconv4,deconv5,\
conv13, conv14, conv15, conv16,conv17,conv18,conv19,conv20, w13, w14,w15,w16,w17,w18,w19,w20

  
  def save(self, checkpoint_dir, step):
    model_name = "SRCNN.model"
    model_dir = "%s_%s" % ("srcnn", self.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = "%s_%s" % ("srcnn", self.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False
