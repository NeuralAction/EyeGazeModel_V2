from __future__ import print_function

import datetime
import math
import os
import random
import shutil
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pympler import refbrowser
from tensorflow.core.framework import attr_value_pb2, graph_pb2, node_def_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import (dtypes, graph_util, ops, tensor_shape,
                                         tensor_util)
from tensorflow.python.layers import utils
from tensorflow.python.ops import array_ops, math_ops, random_ops
from tensorflow.python.platform import flags as flags_lib
from tensorflow.python.platform import tf_logging
from tensorflow.python.tools import optimize_for_inference_lib as optlib
from tensorflow.python.tools import strip_unused_lib

useSELU = False
batchSize = 10

def _variable_with_weight_decay(shape, wd=None):
    with tf.name_scope('wieght'):
        # Determine number of input features from shape
        f_in = np.prod(shape[:-1]) if len(shape) == 4 else shape[0]
        
        # Calculate sdev for initialization according to activation function
        if useSELU:
            sdev = math.sqrt(1 / f_in)
        else:
            sdev = math.sqrt(2 / f_in)
        
        var = tf.Variable(tf.truncated_normal(shape=shape, stddev=sdev))
        if wd is not None:
            weight_decay = tf.reduce_sum(tf.multiply(tf.nn.l2_loss(var), wd))
            tf.add_to_collection('losses', weight_decay)
        return var

managedWeight = []
def convWeight(shape):
    w = _variable_with_weight_decay(shape=shape)
    managedWeight.append(w)
    print("ConvWeihgt:", shape[0]*shape[1]*shape[2]*shape[3])
    return w

def fcWeight(shape, weight_decay = 0.001):
    w = _variable_with_weight_decay(shape=shape, wd=weight_decay)
    managedWeight.append(w)
    print("FcWeihgt:", shape[0]*shape[1])
    return w

def biasWeight(shape):
    with tf.name_scope('bias'):
        w = tf.Variable(tf.constant(0.0, shape=shape, dtype=tf.float32))
        managedWeight.append(w)
        return w

def weightReport():
    c = 0
    fcC = 0
    biasC = 0
    convC = 0
    for i in range(0, len(managedWeight)):
        shp = shape(managedWeight[i])
        shpTotal = 1
        for s in range(0, len(shp)):
            shpTotal *= shp[s]
        c += shpTotal
        if(len(shp)==1):
            biasC+=shpTotal
        elif(len(shp)==2):
            fcC+=shpTotal
        elif(len(shp)==4):
            convC+=shpTotal
    print("Total Weight:", c)
    print("Fc Weight:", fcC)
    print("Bias Weight:", biasC)
    print("Conv Weight:", convC)

def deconv2d(x, W, stride = [2,2], pad='SAME'):
    xShape = shape(x)
    wShape = shape(W)
    outShape = [batchSize, xShape[1]*stride[0], xShape[2]*stride[1], wShape[2]]
    outShape = tf.constant(outShape)
    stridShape = [1, stride[0], stride[1], 1]
    #print(xShape, W, wShape, outShape, stridShape)
    return tf.nn.conv2d_transpose(x, W, outShape, stridShape, padding=pad, name=None)

def deconv2dSingle(pool, weightShape, stride=[2,2], var_dict=None, useactivate=True):
    with tf.name_scope('deconv2dSingle'):
        filterW = weightShape[0]
        filterH = weightShape[1]
        preCh = shape(pool)[3]
        ch = weightShape[2]

        if(var_dict):
            w = var_dict['w']
            b = var_dict['b']
        else:
            w = convWeight([filterW, filterH, ch, preCh])
            b = biasWeight([ch])
            var_dict = {'w' : w, 'b' : b }

        pool = deconv2d(pool, w, stride = stride, pad='SAME') + b
        if(useactivate):
            pool = activate(pool)
        print(pool)

        return pool, var_dict

def conv2d(x, W, stride = 1, pad='SAME'):
    return tf.nn.conv2d(x, W, strides=[1,stride, stride, 1], padding=pad)

def max_pool(x, size=2):
    return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='SAME')

def avg_pool_2x2(x, size=2):
    return tf.nn.avg_pool(x, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='SAME')

#ref. http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
def batch_norm(x, n_out, phase_train, scope='bn'):
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]), trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2])
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        
        mean, var = tf.cond(phase_train, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def tanh(tensor):
    return tf.tanh(tensor)

def sigmoid(tensor):
    return tf.sigmoid(tensor)

def relu(tensor):
    return tf.nn.relu(tensor)

def selu(x):
    with tf.name_scope('selu'):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))

def dropout_selu(x, rate, alpha= -1.7580993408473766, fixedPointMean=0.0, fixedPointVar=1.0, 
                 noise_shape=None, seed=None, name=None, training=False):
    """Dropout to a value with rescaling."""

    def dropout_selu_impl(x, rate, alpha, noise_shape, seed, name):
        keep_prob = 1.0 - rate
        x = ops.convert_to_tensor(x, name="x")
        # if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
        #     raise ValueError("keep_prob must be a scalar tensor or a float in the "
        #                                      "range (0, 1], got %g" % keep_prob)
        keep_prob = ops.convert_to_tensor(keep_prob, dtype=x.dtype, name="keep_prob")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        alpha = ops.convert_to_tensor(alpha, dtype=x.dtype, name="alpha")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        if tensor_util.constant_value(keep_prob) == 1:
            return x

        noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
        random_tensor = keep_prob
        random_tensor += random_ops.random_uniform(noise_shape, seed=seed, dtype=x.dtype)
        binary_tensor = math_ops.floor(random_tensor)
        ret = x * binary_tensor + alpha * (1-binary_tensor)

        a = tf.sqrt(fixedPointVar / (keep_prob *((1-keep_prob) * tf.pow(alpha-fixedPointMean,2) + fixedPointVar)))

        b = fixedPointMean - a * (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
        ret = a * ret + b
        ret.set_shape(x.get_shape())
        return ret

    with tf.name_scope(name, "dropout", [x]) as name:
        return utils.smart_cond(training,
            lambda: dropout_selu_impl(x, rate, alpha, noise_shape, seed, name),
            lambda: array_ops.identity(x))

def dropout(tensor, rate, training, name=None):
    if(useSELU):
        return dropout_selu(tensor, rate, training=training, name=name)
    return tf.nn.dropout(tensor, rate, name=name)

def activate(tensor):
    if(useSELU):
        return selu(tensor)
    return relu(tensor)

def shape(tensor):
    s = tensor.get_shape()
    return tuple([s[i].value for i in range(0, len(s))])

def resBlockPool(tensor, poolsize=2):
    return avg_pool_2x2(tensor, size=poolsize)

def depthConv2d(pool, weight, stride = 1):
    return tf.nn.depthwise_conv2d(pool, weight, [1, stride, stride, 1], padding='SAME')

def depthConv2dSingle(pool, phase_train, useBnorm, kernelShape, chMul = 1, stride = 2, var_dict = None):
    with tf.name_scope('depthConv2D'):
        filterW = kernelShape[0]
        filterH = kernelShape[1]
        preCh = shape(pool)[3]
        ch = preCh * chMul

        if(var_dict is None):
            Wconv = convWeight([filterW, filterH, preCh, chMul])
            Bconv = biasWeight([ch])
            var_dict = {'w' : Wconv, 'b' : Bconv }
        else:
            Wconv = var_dict['w']
            Bconv = var_dict['b']

        pool = depthConv2d(pool, Wconv, stride) + Bconv
        if(useBnorm):
            pool = batch_norm(pool, ch, phase_train)
        pool = activate(pool)
        print(pool)
        
        return pool, var_dict

def mobileNetV2(pool, phase_train, useBnorm, weightShape, poolsize = 2, var_dict = None):
    with tf.name_scope('mobileV2'):
        preCh = shape(pool)[3]
        if(var_dict is None):
            v1 = None
            v2 = None
            v3 = None
        else:
            v1 = var_dict['v1']
            v2 = var_dict['v2']
            v3 = var_dict['v3']
        input = pool
        pool, v1 = conv2dSingle(pool, phase_train, useBnorm, [1, 1, weightShape[2]], poolsize = 1, var_dict = v1)
        pool, v2 = depthConv2dSingle(pool, phase_train, useBnorm, [weightShape[0], weightShape[1]], chMul = 1, stride = poolsize, var_dict = v2)
        pool, v3 = conv2dSingle(pool, phase_train, useBnorm, [1, 1, weightShape[2]], poolsize = 1, var_dict = v3, useAct = False)
        if(poolsize == 1 and weightShape[2] == preCh):
            pool = input + pool
        var_dict = {'v1' : v1, 'v2' : v2, 'v3' : v3 }
        return pool, var_dict
    
def conv2dSingle(pool, phase_train, useBnorm, weightShape, stride = 1, poolsize = 2, var_dict = None, useAct = True):
    with tf.name_scope('conv2dSingle'):
        filterW = weightShape[0]
        filterH = weightShape[1]
        preCh = shape(pool)[3]
        ch = weightShape[2]

        #conv
        if(var_dict):
            W_conv = var_dict['w']
            b_conv = var_dict['b']
        else:
            W_conv = convWeight([filterW, filterH, preCh, ch])
            b_conv = biasWeight([ch])
            var_dict = {'w' : W_conv, 'b' : b_conv}

        h_conv = conv2d(pool, W_conv, stride = stride) + b_conv
        if(useBnorm):
            h_conv = batch_norm(h_conv, ch, phase_train)
        if(useAct):
            h_conv = activate(h_conv)
        h_pool = h_conv
        if not poolsize is 1:
            h_pool = max_pool(h_pool, size = poolsize)
        print(h_pool)

        return h_pool, var_dict

def flat(tensor):
    tShape = shape(tensor)
    return tf.reshape(tensor, [-1, tShape[1] * tShape[2] * tShape[3]])

def fc(tensor, nodeNum, keep_prob, phase_train, name = None, var_dict = None, useactivate = True):
    with tf.name_scope('fc'):
        fcsize = shape(tensor)[1]
        if type(nodeNum) is float:
            nodeNum = int(round(nodeNum * fcsize))
        if(var_dict):
            W_fc = var_dict['w']
            b_fc = var_dict['b']
        else:
            W_fc = fcWeight([fcsize, nodeNum])
            b_fc = biasWeight([nodeNum])
            var_dict = { 'w' : W_fc, 'b': b_fc }
        pool = tf.matmul(tensor, W_fc) + b_fc
        if(useactivate):
            pool = activate(pool)
        pool = dropout(pool, keep_prob, phase_train, name=name)
        return pool, var_dict

def regression(tensor, nodeNum, name=None, var_dict = None):
    tensor_size = shape(tensor)[1]
    if(var_dict):
        W_fc = var_dict['w']
        b_fc = var_dict['b']
    else:
        W_fc = fcWeight([tensor_size, nodeNum])
        b_fc = biasWeight([nodeNum])
        var_dict = {'w' : W_fc,'b' : b_fc}
    return tf.add(tf.matmul(tensor, W_fc), b_fc, name=name), var_dict

def inference(pool, numClass, name = None, var_dict = None):
    fcsize = shape(pool)[1]
    if(var_dict):
        weight = var_dict ['w']
        bias = var_dict['b']
    else:
        weight = fcWeight([fcsize, numClass])
        bias = biasWeight([numClass])
        var_dict = { 'w' : weight, 'b' : bias}
    
    pool = tf.add(tf.matmul(pool, weight), bias)
    pool = tf.nn.softmax(pool, name = name)
    return pool, var_dict

def crossEntropy(y, label):
    with tf.name_scope('crossEntropy'):
        loss = tf.reduce_mean(-tf.reduce_sum(label * tf.log(y), reduction_indices = 1))
    return loss

def correctPrediction(y, label):
    with tf.name_scope('correctPrediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(label, 1))
    return correct_prediction

def inferenceAccuracy(y, label):
    with tf.name_scope('infAcc'):
        acc = tf.reduce_mean(tf.cast(correctPrediction(y, label), "float"))
    return acc

def LoopedExponentialDecayLearningRate(global_step, startRate = 0.001, duration = 1000, decayStep = 100, decayRate = 0.7, durationRate = 1.5):
    global_step = tf.cast(global_step, tf.float32)
    return startRate * tf.pow(decayRate, tf.floor((global_step - tf.floor(global_step / duration) * duration) / decayStep))
    l = tf.floor(\
                tf.log(\
                        (durationRate - 1.0) * global_step / duration + 1.0)\
                         / tf.log(durationRate))
    j = (global_step / duration - \
            (1.0 - tf.pow(durationRate, l)) / \
                (1.0 - durationRate)) / \
            tf.pow(durationRate, l) * duration
    a = startRate * tf.pow(decayRate, tf.floor(j / decayStep))
    return a

def learningRateDecay(learningRate, decayRate, global_step, dataSize, batchSize, rateDecayEpoch):
    decay_r = dataSize / batchSize * rateDecayEpoch
    learning_rate = tf.train.exponential_decay(learningRate, global_step, int(decay_r), decayRate, staircase=True)
    return learning_rate

def gradientClippedMinimize(optimizer, cost, gradMin = -1.0, gradMax = 1.0, global_step = None, useClip = True, var_list = None, var_list_prefix = None, lock_scope = None):
    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    if not var_list_prefix is None and var_list is None:
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, var_list_prefix)
        print('update vars:', var_list)
    if not lock_scope is None:
        if var_list is None:
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        print('found vars:', var_list)
        lock_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, lock_scope)
        print('locked vars:', lock_list)
        new_var_list = []
        for v in var_list:
            locked = False
            for l in lock_list:
                if(v.name == l.name):
                    locked = True
                    break
            if not locked:
                new_var_list.append(v)
        var_list = new_var_list
        print('filtered vars:', var_list)
        
    gvs = optimizer.compute_gradients(cost, var_list = var_list)
    if(useClip):
        with tf.name_scope('gradientClip'):
            capped_gvs = []
            for grad, var in gvs:
                g = grad
                if not g is None:
                    g = tf.clip_by_value(grad, gradMin, gradMax)
                content = (g, var)
                capped_gvs.append(content)
    else:
        capped_gvs = gvs
    train_op = optimizer.apply_gradients(capped_gvs, global_step = global_step)
    return train_op

def weightDecayLoss(loss, name = 'total_loss'):
    with tf.name_scope('weightDecay'):
        tf.add_to_collection('losses', loss)
        totalLoss = tf.add_n(tf.get_collection('losses'), name = name)
        return totalLoss

def getRecentCkpt(targetDir):
    def listDirs(dir):
        for (_, dirs, files) in os.walk(dir):
            return dirs
    def listFiles(dir):
        for _, dirs, files in os.walk(dir):
            return files
    files = listFiles(targetDir)
    ckptNames = []
    for f in files:
        if f.endswith('.meta'):
            cname = os.path.basename(f)[:-5]
            ckptNames.append(cname)
    ckptNames.sort()
    return ckptNames[-1]

class NNModel:
    def __init__(self, keep_prob = None, phase_train = None):
        self.useMobileNet = False
        self.keep_prob = keep_prob
        self.phase_train = phase_train
        self.useBnorm = True
        if(keep_prob is None):
            self.keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
        if(phase_train is None):
            self.phase_train = tf.placeholder(tf.bool, name = 'phase_train')            
        self.var_dict = {}
    
    def getDropRate(self):
        if(useSELU):
            return 0.05
        return 0.75

    def getTestDropRate(self):
        if(useSELU):
            return 0.0
        return 1.0

    def clearDict(self):
        self.var_dict = {}
    
    def getDict(self, name):
        if(name in self.var_dict):
            return self.var_dict[name]
        return None

    def resBlock(self, name, input, filterShape, useBnorm = None, poolsize = 1):
        preShape = shape(input)
        print(preShape)
        with tf.name_scope('resBlock'):
            pool = self.conv2d(name + '_res1', input, filterShape, useBnorm = useBnorm, poolsize = poolsize)
            pool = self.conv2d(name + '_res2', pool, filterShape, useBnorm = useBnorm, poolsize = 1, useAct = False)
            if(poolsize == 1) and (filterShape[2] is preShape[3]):
                pool = input + pool
            pool = activate(pool)
            return pool

    def conv2d(self, name, input, filterShape, useBnorm = None, stride = 1, poolsize = 2, useAct = True, phase_train = None, useMobile = None):
        if(useBnorm is None):
            useBnorm = self.useBnorm
        if not(phase_train):
            phase_train = self.phase_train
        if(useMobile is None):
            useMobile = self.useMobileNet
        v = self.getDict(name)
        if not useMobile:
            pool, v = conv2dSingle(input, phase_train, useBnorm, filterShape, stride = stride, poolsize = poolsize, var_dict = v, useAct = useAct)
        else:
            pool, v = mobileNetV2(input, phase_train, useBnorm, filterShape, poolsize = poolsize, var_dict = v)
        self.var_dict[name] = v
        return pool
    
    def fc(self, name, input, nodeNum, useAct = True, phase_train = None, keep_prob = None):
        v = self.getDict(name)
        if not(phase_train):
            phase_train = self.phase_train
        if not(keep_prob):
            keep_prob = self.keep_prob
        pool, v = fc(input, nodeNum, keep_prob, phase_train, name = None, var_dict = v, useactivate = useAct)
        self.var_dict[name] = v
        return pool
    
    def inference(self, name, input, numClass, opName = None):
        v = self.getDict(name)
        pool, v = inference(input, numClass, name = opName, var_dict = v)
        self.var_dict[name] = v
        return pool

    def reg(self, name, input, nodeNum, opName = None):
        v = self.getDict(name)
        pool, v = regression(input, nodeNum, name = opName, var_dict = v)
        self.var_dict[name] = v
        return pool

    def deconv2d(self, name, input, weightShape, stride = [2, 2], useAct = True):
        v = self.getDict(name)
        pool, v = deconv2dSingle(input, weightShape, stride=stride, var_dict=v, useactivate=useAct)
        self.var_dict[name] = v
        return pool
