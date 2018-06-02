import multiprocessing
import time
import random
import os

import numpy as np
import scipy.ndimage as ndimage
import tensorflow as tf
import skimage

import eyemodel_closeopen
import nn
import util
from tensorflow.python.keras._impl.keras.datasets.cifar10 import load_data
from tensorflow.python.framework import graph_util
from tensorflow.python.tools import optimize_for_inference_lib as optlib
from tensorflow.python.tools import strip_unused_lib

class cifar10:
    def __init__(self):
        self.numClass = 10
        self.batchSize = 256
        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_data()
        self.dataSize = len(self.y_train)
        self.y_train_one_hot = self.onehot(self.y_train, self.numClass)
        self.y_test_one_hot = self.onehot(self.y_test, self.numClass)
        self.pool = util.Parallel()
        self.thread = util.ThreadBuffer()
    
    def onehot(self, idx, numClass):
        return np.asarray([[float(x == idx[i]) for x in range(numClass)] for i in range(len(idx))])
    
    class Proc:
        def __init__(self, isTrain):
            self.isTrain = isTrain
            
        def dataPreProc(self, data, isTrain):
            data = data.astype(float)
            if(isTrain):
                if(random.random() < 0.5):
                    data = np.fliplr(data)
                data = data * (1 + 0.5 * (random.random() - 0.5)) + np.random.random(data.shape) * (random.random() * data.std())
            data = (data - np.average(data)) / np.std(data)
            return data

        def __call__(self, data):
            return self.dataPreProc(data[0], self.isTrain), data[1]

    def getBatch(self, num, data, labels, isTrain = True):
        idx = np.arange(0 , len(data))
        np.random.shuffle(idx)
        idx = idx[:num]
        proc = self.Proc(isTrain)
        poolData = [(data[i], labels[i]) for i in idx]
        poolData = self.pool.map(proc, poolData)
        data_shuffle = [poolData[i][0] for i in range(num)]
        labels_shuffle = [poolData[i][1] for i in range(num)]
        del proc
        return np.asarray(data_shuffle), np.asarray(labels_shuffle)
    
    def _nextBatch(self, isTrain):
        x = self.x_train
        y = self.y_train_one_hot
        if not isTrain:
            x = self.x_test
            y = self.y_test_one_hot
        return self.getBatch(self.batchSize, x, y, isTrain)

    def nextBatch(self, isTrain = True):
        return self.thread.get(self._nextBatch, [isTrain])

    def close(self):
        self.pool.close()
        self.thread.close()

class TransferHelper(nn.NNModel):
    def __init__(self):
        self.keep_prob = tf.placeholder(tf.float32, name = 'transfer_keep_prob')
        self.phase_train = tf.placeholder(tf.bool, name = 'transfer_phase_train')     
        super(TransferHelper, self).__init__(keep_prob = self.keep_prob, phase_train = self.phase_train)
        self.data = cifar10()
    
    def close(self):
        self.data.close()

    def pretrain(self, sess, input, inputShape, output, phase_train = None, keep_prob = None, dropRate = 0.05, testDropRate = 0.0, targetAcc = 0.975, maxEphoc = 100, maxTime = 360000):
        #input should image. output should last fc output of CNN
        self.clearDict()
        self.startTime = time.time()
        self.inputWidth = inputShape[0]
        self.inputHeight = inputShape[1]
        
        label = tf.placeholder_with_default([[0.0 for i in range(self.data.numClass)]], shape=[None, self.data.numClass])

        pool = self.inference('transferInf', output, self.data.numClass)
        loss = nn.weightDecayLoss(nn.crossEntropy(pool, label), name = None)
        accuracy = nn.inferenceAccuracy(pool, label)
        tf.summary.scalar('transferLoss', loss)
        tf.summary.scalar('transferAcc', accuracy)

        globalStep = tf.Variable(0, trainable = False)
        learningRate = nn.learningRateDecay(0.001, 0.7, globalStep, self.data.dataSize, self.data.batchSize, 10)
        optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
        trainStep = nn.gradientClippedMinimize(optimizer, loss, global_step = globalStep)
        tf.summary.scalar('transferLR', learningRate)

        sess.run(tf.global_variables_initializer())
        step = 0
        keepRun = True
        fps = util.FpsCounter()
        while keepRun:
            step += 1
            ephoc = step * self.data.batchSize / self.data.dataSize
            x, y = self.data.nextBatch()
            fetch = sess.run([trainStep, loss, accuracy, learningRate], feed_dict = {input : x, label : y, phase_train : True, keep_prob : dropRate})
            fps.add(self.data.batchSize)
            fetch = fetch[1:]
            if(step % 10 == 0):
                x_test, y_test = self.data.nextBatch(False)
                tfetch = sess.run([loss, accuracy], feed_dict = {input : x_test, label : y_test, phase_train : True, keep_prob : testDropRate})
                print('[transfer-training] step:', step, 'ephoc:', str(int(ephoc)) + '(%0.2f%%)' % (ephoc % 1 * 100), 'fetch:', fetch, 'tfetch:', tfetch, 'data/s:', fps.fps())
                elapsed = time.time() - self.startTime
                if fetch[1] > targetAcc or ephoc > maxEphoc or elapsed > maxTime:
                    print('cifar10 train is finished', 'acc:', fetch[1], 'ephoc:', int(ephoc), 'time:', elapsed)
                    return

class ModelSaver:
    def __init__(self, parentPath, checkpointName, useBnorm = False, inputNodes = None, inputNodesTypes = None, outputNodes = None):
        self.parentPath = parentPath
        self.checkpointName = checkpointName
        self.useBnorm = useBnorm
        self.inputNodes = inputNodes
        self.inputNodesTypes = inputNodesTypes
        self.outputNodes = outputNodes
        self.load()
    
    def load(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        print("importing...")
        saver = tf.train.import_meta_graph(os.path.join(self.parentPath, self.checkpointName + '.meta'))
        print("restoring...")
        saver.restore(self.sess, os.path.join(self.parentPath, self.checkpointName))
        graph = self.sess.graph
        # self.inputLeft = graph.get_tensor_by_name('input_left:0')
        # self.inputRight = graph.get_tensor_by_name('input_right:0')
        # self.inputFace = graph.get_tensor_by_name('input_face:0')
        # self.keep_prob = graph.get_tensor_by_name('keep_prob:0')
        # self.output = graph.get_tensor_by_name('output:0')
        # self.phase_train = graph.get_tensor_by_name('phase_train:0')
    
    def freeze(self):
        gd = self.sess.graph.as_graph_def()
        print("convt..")
        for node in gd.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']
        print("const...")
        gd = graph_util.convert_variables_to_constants(self.sess, gd, self.outputNodes)

        optlib.ensure_graph_is_valid(gd)
        input_node_names = self.inputNodes
        output_node_names = self.outputNodes
        placeholder_type_enum = self.inputNodesTypes
        for i in range(len(placeholder_type_enum)):
            placeholder_type_enum[i] = placeholder_type_enum[i].as_datatype_enum
        print("strip...")
        gd = strip_unused_lib.strip_unused(gd, input_node_names, output_node_names, placeholder_type_enum)
        optlib.ensure_graph_is_valid(gd)
        filename = 'frozen ' + util.getTimeStamp() + '.pb'
        tf.train.write_graph(gd, self.parentPath, filename, as_text=False)
        return os.path.join(self.parentPath, filename)

class Model(nn.NNModel):
    def __init__(self, dataSize, batchSize):
        super(Model, self).__init__()
        nn.useSELU = False
        self.useBnorm = True
        self.input = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name = 'input')
        self.label = tf.placeholder_with_default([[0.0, 0.0]], shape=[None, 2])
        self.outputCNN = self.buildCNN(self.input)
        self.output = self.buildInf(self.outputCNN)

        self.loss = nn.weightDecayLoss(nn.crossEntropy(self.output, self.label), name = None)
        self.accuracy = nn.inferenceAccuracy(self.output, self.label)

        globalStep = tf.Variable(0, trainable = False)
        self.learningRate = nn.learningRateDecay(0.001, 0.7, globalStep, dataSize, batchSize, 5)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learningRate)
        self.trainStep = nn.gradientClippedMinimize(optimizer, self.loss, global_step = globalStep, lock_scope = 'lock')

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('learningRate', self.learningRate)

    def buildCNN(self, input):
        n = util.NameGenerator('cnn')
        with tf.name_scope('lock'):
            pool = self.conv2d(n.new(), input, [3, 3, 32], poolsize = 1)
            pool = self.conv2d(n.new(), pool, [3, 3, 32])
            pool = self.conv2d(n.new(), pool, [3, 3, 64], poolsize = 1)
            pool = self.conv2d(n.new(), pool, [3, 3, 64])
            pool = self.conv2d(n.new(), pool, [3, 3, 128], poolsize = 1)
            pool = self.conv2d(n.new(), pool, [3, 3, 128], poolsize = 1)
        pool = self.conv2d(n.new(), pool, [3, 3, 128])
        pool = nn.flat(pool)
        pool = self.fc(n.new(), pool, 384)
        return pool
    
    def buildInf(self, outputCNN):
        n = util.NameGenerator('inference')
        pool = self.inference(n.new(), outputCNN, 2, opName = 'output')
        return pool

    def optimize(self, sess, input, label):
        feed_dict = { self.input : input, self.label : label, self.phase_train : True, self.keep_prob : self.getDropRate()}
        fetch = sess.run([self.trainStep, self.loss, self.accuracy, self.learningRate], feed_dict = feed_dict)
        return fetch[1:]

    def forward(self, sess, input, label):
        feed_dict = { self.input : input, self.label : label, self.phase_train : True, self.keep_prob : self.getTestDropRate()}
        fetch = sess.run([self.trainStep, self.loss, self.accuracy], feed_dict = feed_dict)
        return fetch[1:]

class Dataset:
    def __init__(self):
        self.loadData()
        self.thread = util.ThreadBuffer()
    
    def loadData(self):
        p = multiprocessing.Pool(processes=14)

        basedir = "C:\\Library\\koi 2017\\Source\\OpenDataset\\"
        dataListOpen = [basedir+"open1\\left\\",
                        basedir+"open1\\right\\",
                        basedir+"open2\\left\\",
                        basedir+"open2\\right\\",
                        basedir+"open3\\left\\",
                        basedir+"open3\\right\\",]
        
        dataListClose = [basedir+"close1\\left\\",
                        basedir+"close1\\right\\",
                        basedir+"close2\\left\\",
                        basedir+"close2\\right\\",
                        basedir+"close3\\left\\",
                        basedir+"close3\\right\\",
                        basedir+"close4\\left\\",
                        basedir+"close4\\right\\",]

        print("READ OPEN DATA")
        dataOpen = eyemodel_closeopen.decodeData(dataListOpen, p)
        dataOpen.imagesize = 32
        dataOpen.rotate = 360
        dataOpen.randpad = 0.1

        print("READ CLOSE DATA")
        dataClose = eyemodel_closeopen.decodeData(dataListClose, p)
        dataClose.imagesize = dataOpen.imagesize
        dataClose.rotate = dataOpen.rotate
        dataClose.randpad = dataOpen.randpad

        datatest = eyemodel_closeopen.decodeData([ basedir + "valid\\", basedir+"open3\\right\\", basedir+"close4\\right\\" ], p)
        datatest.imagesize = dataOpen.imagesize
        datatest.rotate = dataOpen.rotate

        self.dataOpen = dataOpen
        self.dataClose = dataClose
        self.dataTest = datatest
        self.count = dataOpen.size + dataClose.size
    
    def _batch(self, count, isTrain):
        if not isTrain:
            tbatch_img, tbatch_label = self.dataTest.batch(count, randomize = False)
            return tbatch_img, tbatch_label
        batch_img_open, batch_label_open = self.dataOpen.batch(int(round(count / 2)))
        batch_img_close, batch_label_close = self.dataClose.batch(int(round(count / 2)))
        batch_img = np.concatenate((batch_img_open, batch_img_close), axis=0)
        batch_label = np.concatenate((batch_label_open, batch_label_close), axis=0)
        return batch_img, batch_label

    def batch(self, count, isTrain = True):
        return self.thread.get(self._batch, [count, isTrain])

def freeze(sessName):
    targetDir = './temp/' + sessName
    ckpt = nn.getRecentCkpt(targetDir)
    print(ckpt)
    saver = ModelSaver(targetDir, ckpt, useBnorm = True, 
        inputNodes = [ 'input', 'phase_train', 'keep_prob' ], 
        inputNodesTypes = [ tf.float32, tf.bool, tf.float32 ], 
        outputNodes = [ 'output' ])
    saver.freeze()

def main():
    batchCount = 128
    lastEphoc = 0
    step = 0
    fps = util.FpsCounter()
    timestamp = util.getTimeStamp()
    testDir = './temp/blink-test ' + timestamp

    data = Dataset()
    transfer = TransferHelper()
    model = Model(dataSize = data.count, batchSize = batchCount)
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        sess.run(tf.global_variables_initializer())

        transfer.pretrain( \
            sess, model.input, [32, 32, 3], model.outputCNN,
            phase_train = model.phase_train,
            keep_prob = model.keep_prob,
            maxEphoc = 500,
            targetAcc = 1,
            dropRate = model.getDropRate(),
            testDropRate = model.getTestDropRate())
        transfer.close()

        while True:
            step += 1
            ephoc = step * batchCount / data.count
            fps.add(batchCount)
            batchImg, batchLabel = data.batch(batchCount)
            fetch = model.optimize(sess, batchImg, batchLabel)
            del batchImg, batchLabel
            if(step % 10 == 0):
                tbatchImg, tbatchLabel = data.batch(batchCount, False)
                tfetch = model.forward(sess, tbatchImg, tbatchLabel)
                print('step:', step, 'ephoc:', int(ephoc), '(%0.2f%%)' % (ephoc % 1 * 100) , 'fetch:', fetch, 'tfetch:', tfetch, 'data/s:', fps.fps())
                del tbatchImg, tbatchLabel
            if(lastEphoc != int(ephoc)):
                lastEphoc = int(ephoc)
                ckpt_path = saver.save(sess, testDir + '/model.ckpt', global_step = step)
                print('chpt saved:', ckpt_path)

if __name__ == '__main__':
    main()
    freeze('blink-test 03-31_23-58-12')
