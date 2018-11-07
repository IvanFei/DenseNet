# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 10:16:26 2018

@author: PISME_Public01

"""
"""
方差缩放初始化？
https://zhuanlan.zhihu.com/p/38315135
"""

import numpy as np
import tensorflow as tf
from tqdm import trange, tqdm
from PIL import Image
import matplotlib.pyplot as plt
import os
import shutil

from tensorflow.examples.tutorials.mnist import input_data

class DenseNet(object):
    def __init__(self, datasets, growth_rate, depth, 
                 total_blocks, keep_prob, weight_decay, nesterov_momentum, model_type, dataset_name,
                 should_save_logs, should_save_model, 
                 renew_logs=False,
                 reduction=1.0,
                 bc_mode=False,
                 **kwargs):
        
        self.datasets = datasets
        if dataset_name == 'fashion-mnist':
            self.data_shape = (28, 28, 1)
            self.n_classes = 10
        elif dataset_name == 'mnist':
            self.data_shape = (28, 28, 1)
            self.n_classes = 10
        
        self.depth = depth
        self.growth_rate = growth_rate
        self.first_output_features = growth_rate * 2
        self.total_blocks = total_blocks
        self.layers_per_block = (depth - (total_blocks + 1)) // total_blocks
        self.bc_mode = bc_mode
        self.reduction = reduction
        
        if not bc_mode:
            print(f"Build {model_type} model with {self.total_blocks} blocks, "
                  f"{self.layers_per_block} composite layers each.")
        else:
            self.layers_per_block = self.layers_per_block // 2
            print(f"Build {model_type} model with {self.total_blocks} blocks, "
                  f"{self.layers_per_block} bottleneck layers and {self.layers_per_block} composite layers each.")
        
        print(f"Reduction at transition layers: {self.reduction}")
        
        self.keep_prob = keep_prob
        self.weight_decay = weight_decay
        self.nesterov_momentum = nesterov_momentum
        self.model_type = model_type
        self.dataset_name = dataset_name
        self.should_save_logs = should_save_logs
        self.should_save_model = should_save_model
        self.renew_logs = renew_logs
        self.batches_step = 0
        
        self._define_inputs()
        self._build_graph()
        self._initialize_session()
        self._count_trainable_params()
    
    def _initialize_session(self):
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        logswriter = tf.summary.FileWriter
        
        self.saver = tf.train.Saver()
        self.summary_weiter = logswriter(self.logs_path)
        
    def _count_trainable_params(self):
        
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print(f"Total training params: {total_parameters / 1e6}M")
    
    @property
    def save_path(self):
        
        try:
            save_path = self._save_path
        except AttributeError:
            save_path = f"saves/{self.model_identifier}"
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, "model.ckpt")
            self._save_path = save_path
        return save_path
    
    @property
    def logs_path(self):
        
        try:
            logs_path = self._logs_path
        except AttributeError:
            logs_path = f"logs/{self.model_identifier}"
            if self.renew_logs:
                shutil.rmtree(logs_path, ignore_errors=True)
            os.makedirs(logs_path, exist_ok=True)
            self._logs_path = logs_path
        return logs_path
    
    @property
    def model_identifier(self):
        
        return f"{self.model_type}_growth_rate={self.growth_rate}_depth={self.depth}_dataset_{self.dataset_name}"
    
    def save_model(self, global_step=None):
        
        self.saver.save(self.sess, self.save_path, global_step=global_step)
    
    def load_model(self):
        
        try:
            self.saver.restore(self.sess, self.save_path)
        except Exception as e:
            raise IOError("Failed to load model "
                          f"from save path: {self.save_path}")
        self.saver.restore(self.sess, self.save_path)
        print(f"Successfully load model from save path: {self.save_path}")
    
    def log_loss_accuracy(self, loss, accuracy, epoch, prefix, should_print=True):
        
        if should_print:
            print(f"mean cross_entropy: {loss}, mean accuracy: {accuracy}")
        summary = tf.Summary(value=[
                tf.Summary.Value(
                        tag=f'loss_{prefix}', simple_value=float(loss)),
                tf.Summary.Value(
                        tag=f'accuracy_{prefix}', simple_value=float(accuracy))
                ])
        self.summary_weiter.add_summary(summary)
        
    def _define_inputs(self):
        
        shape = [None]
        shape.extend(self.data_shape)
        self.images = tf.placeholder(
                tf.float32, 
                shape=shape,
                name='input_images')
        
        self.labels = tf.placeholder(
                tf.float32,
                shape=[None, self.n_classes],
                name='labels')
        
        self.learning_rate = tf.placeholder(
                tf.float32,
                shape=[],
                name='learning_rate')
        
        self.is_training = tf.placeholder(
                tf.bool,
                shape=[])
        
    def composite_function(self, _input, out_features, kernel_size=3):
        
        with tf.variable_scope("composite_function"):
            # BN
            output = self.batch_norm(_input)
            # ReLu
            output = tf.nn.relu(output)
            # convolution
            output = self.conv2d(
                    output, out_features=out_features, kernel_size=kernel_size)
            # dropout
            output = self.dropout(output)
        return output
            
    def bottleneck(self, _input, out_features):
        
        with tf.variable_scope("bottleneck"):
            output = self.batch_norm(_input)
            output = tf.nn.relu(output)
            inter_features = out_features * 4
            output = self.conv2d(
                    output, out_features=inter_features, kernel_size=1,
                    padding='VALID')
            output = self.dropout(output)
        return output
    
    def add_internal_layer(self, _input, growth_rate):
        
        if not self.bc_mode:
            comp_out = self.composite_function(
                    _input, out_features=growth_rate, kernel_size=3)
        elif self.bc_mode:
            bottleneck_out = self.bottleneck(
                    _input, out_features=growth_rate)
            comp_out = self.composite_function(
                    bottleneck_out, out_features=growth_rate, kernel_size=3)
            
        output = tf.concat(axis=3, values=(_input, comp_out))
        return output
        
    def add_block(self, _input, growth_rate, layers_per_block):
        
        output = _input
        for layer in range(layers_per_block):
            with tf.variable_scope(f"layer_{layer}"):
                output = self.add_internal_layer(
                        output, growth_rate)
        return output
    
    def trainsition_layer(self, _input):
        
        out_features = int(int(_input.get_shape()[-1]) * self.reduction)
        output = self.composite_function(
                _input, out_features, kernel_size=1)
        output = self.avg_pool(_input, 2)
        return output
    
    def transition_layer_to_classes(self, _input):
        
        # BN
        output = self.batch_norm(_input)
        # ReLu
        output = tf.nn.relu(output)
        # average pooling by channel
        last_pool_kernel = int(output.get_shape()[-2])
        output = self.avg_pool(output, k=last_pool_kernel)
        # FC
        features_total = int(output.get_shape()[-1])
        output = tf.reshape(output, [-1, features_total])
        W  = self.weight_variable_xavier(
                [features_total, self.n_classes], name='W')
        bias = self.bias_variable([self.n_classes])
        logits = tf.matmul(output, W) + bias
        return logits
        
    def conv2d(self, _input, out_features, kernel_size, 
               strides=[1, 1, 1, 1], padding='SAME'):
        
        in_features = int(_input.get_shape()[-1])
        kernel = self.weight_variable_msra(
                [kernel_size, kernel_size, in_features, out_features], name='kernel1')
        output = tf.nn.conv2d(_input, kernel, strides, padding)
        return output
    
    def avg_pool(self, _input, k):
        
        ksize = [1, k, k, 1]
        strides = [1, k, k, 1]
        padding = 'VALID'
        output = tf.nn.avg_pool(_input, ksize, strides, padding)
        return output
    
    def batch_norm(self, _input):
        
        output = tf.contrib.layers.batch_norm(
                _input, scale=True, is_training=self.is_training,
                updates_collections=None)
        return output
    
    def dropout(self, _input):
        
        if self.keep_prob < 1:
            output = tf.cond(
                    self.is_training,
                    lambda: tf.nn.dropout(_input, self.keep_prob),
                    lambda: _input
            )
        else:
            output = _input
        return output
    
    
    def weight_variable_msra(self, shape, name):
        
        return tf.get_variable(
                name=name, 
                shape=shape,
                initializer=tf.contrib.layers.variance_scaling_initializer())
    
    def weight_variable_xavier(self, shape, name):
        
        return tf.get_variable(
                name=name,
                shape=shape,
                initializer=tf.contrib.layers.xavier_initializer())
    
    def bias_variable(self, shape, name='bias'):
        
        inital = tf.constant(0.0, shape=shape)
        return tf.get_variable(name, initializer=inital)
    
    
    def _build_graph(self):
        
        growth_rate = self.growth_rate
        layers_per_block = self.layers_per_block
        # first 
        with tf.variable_scope("Initial_convolution"):
            output = self.conv2d(
                    self.images,
                    out_features=self.first_output_features,
                    kernel_size=3)
        
        for block in range(self.total_blocks):
            with tf.variable_scope(f"Block_{block}"):
                output = self.add_block(
                        output, 
                        growth_rate,
                        layers_per_block)
            if block != self.total_blocks - 1:
                with tf.variable_scope(f"Transition_after_block_{block}"):
                    output = self.trainsition_layer(output)
                
        with tf.variable_scope("Transition_to_classes"):
            logits = self.transition_layer_to_classes(output)
        prediction = tf.nn.softmax(logits)
        
        # Losses
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=self.labels))
        self.cross_entropy = cross_entropy
        l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        
        #optimizer = tf.train.MomentumOptimizer(
        #        self.learning_rate, self.nesterov_momentum, use_nesterov=True)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_step = optimizer.minimize(
                cross_entropy + l2_loss * self.weight_decay)
        
        correct_prediction = tf.equal(
                tf.arg_max(prediction, 1), 
                tf.arg_max(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
    def train_all_epochs(self, train_params):
        
        n_epochs = train_params['n_epochs']
        learning_rate = train_params['initial_learning_rate']
        batch_size = train_params['batch_size']
        reduce_lr_epoch_1 = train_params['reduce_lr_epoch_1']
        reduce_lr_epoch_2 = train_params['reduce_lr_epoch_2']
        for epoch in trange(1, n_epochs + 1):
            if epoch == reduce_lr_epoch_1 or epoch == reduce_lr_epoch_2:
                learning_rate = learning_rate / 10
                print(f"Decrease learning rate, new lr = {learning_rate}")
            
            print("Training...")
            loss, acc = self.train_one_epoch(
                    self.datasets.train, batch_size, learning_rate)
            if self.should_save_logs:
                self.log_loss_accuracy(loss, acc, epoch, prefix='train')
            
            if train_params.get('validation_set', False) and epoch % 20==0:
                print("Validation...")
                loss, acc =self.test(self.datasets.validation, batch_size)
                if self.should_save_logs:
                    self.log_loss_accuracy(loss, acc, epoch, prefix='valid')
            
            if self.should_save_model:
                self.save_model()
    
    
    def train_one_epoch(self, data, batch_size, learning_rate):
        
        num_examples = data.num_examples
        total_loss = []
        total_accuracy = []
        for i in range(num_examples // batch_size):
            batch = data.next_batch(batch_size)
            images, labels = batch
            shape = [-1]
            shape.extend(self.data_shape)
            images = images.reshape(shape)
            feed_dict = {
                    self.images: images, 
                    self.labels: labels,
                    self.learning_rate: learning_rate,
                    self.is_training: True,
            }
            fetches = [self.train_step, self.cross_entropy, self.accuracy]
            result = self.sess.run(fetches, feed_dict=feed_dict)
            _, loss, accuracy = result
            total_loss.append(loss)
            total_accuracy.append(accuracy)
            if self.should_save_logs:
                self.batches_step += 1
                self.log_loss_accuracy(
                        loss, accuracy, self.batches_step, prefix='per_batch',
                        should_print=False)
        mean_loss = np.mean(total_loss)
        mean_accuracy = np.mean(total_accuracy)
        return mean_loss, mean_accuracy
            
    def test(self, data, batch_size):
        
        num_examples = data.num_examples
        total_loss = []
        total_accuracy = []
        for i in range(num_examples // batch_size):
            batch = data.next_batch(batch_size)
            images, labels = batch
            shape = [-1]
            shape.extend(self.data_shape)
            images = images.reshape(shape)
            feed_dict = {
                    self.images: images,
                    self.labels: labels,
                    self.is_training: False
            }
            fetches = [self.cross_entropy, self.accuracy]
            loss, accuracy = self.sess.run(
                    fetches, feed_dict=feed_dict)
            total_loss.append(loss)
            total_accuracy.append(accuracy)
        mean_loss = np.mean(total_loss)
        mean_accuracy = np.mean(total_accuracy)
        return mean_loss, mean_accuracy
            
if __name__ == '__main__':
    
    # data = input_data.read_data_sets('data/fashion')
    data = input_data.read_data_sets('data/fashion', source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/', one_hot=True)
    
    params = {
            'datasets': data,
            'growth_rate': 12,
            'depth': 40,
            'total_blocks': 3,
            'keep_prob': 0.8,
            'weight_decay': 0.0005,
            'nesterov_momentum': 1 ,
            'model_type': 'DenseNet_BC',
            'dataset_name': 'fashion-mnist',
            'should_save_logs': True,
            'should_save_model': True}
    model = DenseNet(
            params['datasets'], params['growth_rate'], params['depth'],
            params['total_blocks'], params['keep_prob'], params['weight_decay'],
            params['nesterov_momentum'], params['model_type'], params['dataset_name'],
            params['should_save_logs'], params['should_save_model'], reduction=0.5, renew_logs=True, bc_mode=True)
    
    train_params = {
            'n_epochs': 300,
            'initial_learning_rate': 0.1,
            'batch_size': 32,
            'reduce_lr_epoch_1': 100,
            'reduce_lr_epoch_2': 200,
            'validation_set': True}
    
    model.train_all_epochs(train_params)       
        