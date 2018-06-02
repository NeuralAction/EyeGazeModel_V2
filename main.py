import codecs
import gc
import math
import os
import random
import signal
import sys
import time
from multiprocessing import Pool

import numpy as np
import scipy
import tensorflow as tf
from scipy import misc, ndimage
from skimage import transform
from tensorflow.python.framework import graph_util
from tensorflow.python.tools import optimize_for_inference_lib as optlib
from tensorflow.python.tools import strip_unused_lib
import matplotlib
import matplotlib.pyplot as plt
import concurrent.futures

import nn
from util import *

class Dataset:
    def __init__(self, paths, eyeSize, faceSize):
        self.eyeSize = eyeSize
        self.faceSize = faceSize
        self.count = 0
        self.testPercent = 0.1
        self.datas = []
        self.train = []
        self.test = []

        for i in paths:
            self.addPath(i)
        self.suffleTest()
        self.pool = Parallel(numWorkers = 16)
        self.threadTrain = ThreadBuffer()
        self.threadTest = ThreadBuffer()
        
        print('readed:', self.count, 'train:', len(self.train), 'test:', len(self.test))

    def suffleTest(self):
        self.train = []
        self.test = []
        for i in self.datas:
            if(random.random() >= self.testPercent):
                self.train.append(i)
            else:
                self.test.append(i)
    
    def addPath(self, parentpath):
        walkpath = os.path.join(parentpath, "left")
        for (path, dir, files) in os.walk(walkpath):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext == ".jpg":
                    filepath = os.path.join(os.path.join(parentpath, "left"), filename)
                    imgpath_left = filepath
                    imgpath_right = imgpath_left.replace("left", "right")
                    imgpath_face = imgpath_left.replace("left", "face")
                    if os.path.exists(imgpath_face) and os.path.exists(imgpath_left) and os.path.exists(imgpath_right):
                        self.datas.append(filepath)
        self.count = len(self.datas)

    def getRandom(self, content, count):
        out = []
        for i in range(count):
            ind = int(random.random() * len(content))
            out.append(content[ind])
        return out

    class Proc:
        def __init__(self, eyeSize, faceSize, randomize):
            self.randomize = randomize
            self.eyeSize = eyeSize
            self.faceSize = faceSize
            self.randmul = 0.3
            self.randadd = 15

        def decodeImage(self, filename, size, randomize):
            #read
            img = misc.imread(filename)
            imgr = misc.imresize(img, (size, size))
            imgr = imgr.astype('float32', copy = False)
            del img
            #random
            if(randomize):
                rand_mul = (random.random() * self.randmul - self.randmul / 2) + 1
                rand_add = random.random() * (self.randadd * 2) - self.randadd
                np.multiply(imgr, rand_mul, out = imgr)
                np.add(imgr, rand_add, out = imgr)
                rnd = np.random.random(imgr.shape)
                np.multiply(rnd, random.random() * imgr.std() * 0.6, out = rnd)
                np.add(imgr, rnd, out=imgr)
                np.clip(imgr, 0, 255, out=imgr)
                del rnd
            np.subtract(imgr, np.average(imgr), out=imgr)
            std = np.std(imgr)
            if((abs(std) < 0.01) or math.isnan(std) or math.isinf(std)):
                std = 63.5
            np.divide(imgr, std, out=imgr)
            return imgr
        
        def parseLabel(self, filename):
            name = os.path.splitext(filename)[0]
            nameSpl = name.split(',')
            for i in range(1, len(nameSpl)):
                nameSpl[i] = float(nameSpl[i])
            rod1 = nameSpl[1]
            rod2 = nameSpl[2]
            rod3 = nameSpl[3]
            rt = 1.0 / rod3
            rod1 *= rt
            rod2 *= rt
            return [rod1, rod2]
        
        def __call__(self, fileL):
            fileR = fileL.replace("left", "right")
            fileF = fileL.replace("left", "face")
            imgL = self.decodeImage(fileL, self.eyeSize, self.randomize)
            imgR = self.decodeImage(fileR, self.eyeSize, self.randomize)
            imgF = self.decodeImage(fileF, self.faceSize, self.randomize)
            label = self.parseLabel(fileL)
            del fileR, fileF, fileL
            return imgL, imgR, imgF, label

    def internalBatch(self, count, files, randomize):
        labels = np.empty(shape=[count, 2], dtype='float32')
        left = np.empty(shape=[count, self.eyeSize, self.eyeSize, 3], dtype='float32')
        right = np.empty(shape=[count, self.eyeSize, self.eyeSize, 3], dtype='float32')
        face = np.empty(shape=[count, self.faceSize, self.faceSize, 3], dtype='float32')
        ind = 0
        proc = self.Proc(self.eyeSize, self.faceSize, randomize)
        result = self.pool.map(proc, files)
        for i in result:
            left[ind, :, :, :] = i[0]
            right[ind, :, :, :] = i[1]
            face[ind, :, :, :] = i[2]
            labels[ind, : ] = i[3]
            ind += 1
        del result[:]
        del result, proc
        data = ModelBatchData(face, left, right, label = labels)
        return data

    def _batch(self, count):
        files = self.getRandom(self.train, count)
        ret = self.internalBatch(count, files, True)
        del files[:]
        del files
        return ret

    def batch(self, count):
        return self.threadTrain.get(self._batch, [count])

    def _batchTest(self, count):
        files = self.getRandom(self.test, count)
        ret = self.internalBatch(count, files, False)
        del files[:]
        del files
        return ret

    def batchTest(self, count):
        return self.threadTest.get(self._batchTest, [count])

    def close(self):
        self.pool.close()
    
class ModelBatchData:
    def __init__(self, face, left, right, label = None):
        self.face = face
        self.left = left
        self.right = right
        self.label = label
    
    def dispose(self):
        del self.face
        del self.left
        del self.right
        if(not self.label is None):
            del self.label
        
class Model(nn.NNModel):
    def __init__(self, faceSize = 60, eyeSize = 60, dataSize = 10000, batchSize = 100, useRateDecay = True, rateDecayEpoch = 5, useSELU = False, useSwitching = False, useWeightDecay = True, useMobileNet = False):
        super(Model, self).__init__()
        self.faceSize = faceSize
        self.eyeSize = eyeSize
        self.dataSize = dataSize
        self.batchSize = batchSize
        self.useRateDecay = useRateDecay
        self.rateDecayEpoch = rateDecayEpoch
        self.useSELU = useSELU
        nn.useSELU = useSELU
        self.useMobileNet = useMobileNet
        self.useBnorm = not useSELU
        self.dropRate = 0.7
        self.testDropRate = 1.0
        if(self.useSELU):
            self.dropRate =  0.05
            self.testDropRate = 0.0
        self.step = 0
        self.useSwitching = useSwitching
        self.useWeightDecay = useWeightDecay

        self.inputLeft = tf.placeholder(tf.float32, shape = [None, self.eyeSize, self.eyeSize, 3], name = 'input_left')
        self.inputRight = tf.placeholder(tf.float32, shape = [None, self.eyeSize, self.eyeSize, 3], name = 'input_right')
        self.inputFace = tf.placeholder(tf.float32, shape = [None, self.faceSize, self.faceSize, 3], name = 'input_face')
        self.inputLabel = tf.placeholder(tf.float32, shape = [None, 2])
        
        self.buildModel()
    
    def buildModel(self):
        self.featureFace = self.buildFace(self.inputFace)
        self.featureEyes = self.buildEyes(self.inputLeft, self.inputRight)
        self.output = self.buildRegression(self.featureFace, self.featureEyes)
        self.loss = self.buildLoss(self.output, self.inputLabel)
        self.buildTrainer(self.loss)
        
    def buildFace(self, pool):
        n = NameGenerator('face')
        pool = self.conv2d(n.new(), pool, [3, 3, 32], poolsize = 1, useMobile = False)
        pool = self.conv2d(n.new(), pool, [3, 3, 32], useMobile = False) #16
        pool = self.conv2d(n.new(), pool, [3, 3, 64], poolsize = 1)
        pool = self.conv2d(n.new(), pool, [3, 3, 64], poolsize = 1)
        pool = self.conv2d(n.new(), pool, [3, 3, 64], poolsize = 1)
        pool = self.conv2d(n.new(), pool, [3, 3, 128], poolsize = 1)
        pool = self.conv2d(n.new(), pool, [3, 3, 128], poolsize = 1)
        pool = self.conv2d(n.new(), pool, [3, 3, 128], poolsize = 1)
        pool = self.conv2d(n.new(), pool, [3, 3, 128], poolsize = 1)
        pool = self.conv2d(n.new(), pool, [3, 3, 128]) #8
        pool = self.conv2d(n.new(), pool, [3, 3, 256], poolsize = 1)
        pool = self.conv2d(n.new(), pool, [3, 3, 256], poolsize = 1)
        pool = self.conv2d(n.new(), pool, [3, 3, 256]) #4
        pool = nn.flat(pool)
        pool = self.fc(n.new(), pool, 96)
        pool = self.fc(n.new(), pool, 48)
        return pool
    
    def buildEye(self, pool):
        n = NameGenerator('eye')
        pool = self.conv2d(n.new(), pool, [3, 3, 24], poolsize = 1, useMobile = False)
        pool = self.conv2d(n.new(), pool, [3, 3, 24], useMobile = False) #32
        pool = self.conv2d(n.new(), pool, [3, 3, 32], poolsize = 1)
        pool = self.conv2d(n.new(), pool, [3, 3, 32], poolsize = 1)
        pool = self.conv2d(n.new(), pool, [3, 3, 32]) #16
        pool = self.resBlock(n.new(), pool, [3, 3, 64])
        pool = self.resBlock(n.new(), pool, [3, 3, 64])
        pool = self.resBlock(n.new(), pool, [3, 3, 64])
        pool = self.resBlock(n.new(), pool, [3, 3, 64])
        pool = self.resBlock(n.new(), pool, [3, 3, 64])
        pool = self.resBlock(n.new(), pool, [3, 3, 64])
        pool = self.conv2d(n.new(), pool, [3, 3, 128]) #8
        pool = self.resBlock(n.new(), pool, [3, 3, 128])
        pool = self.resBlock(n.new(), pool, [3, 3, 128])
        pool = self.conv2d(n.new(), pool, [3, 3, 128]) #4
        return pool

    def buildEyes(self, inputLeft, inputRight):
        n = NameGenerator('eyes')
        poolLeft = self.buildEye(inputLeft)
        poolRight = self.buildEye(inputRight)
        pool = tf.concat([nn.flat(poolLeft), nn.flat(poolRight)], 1)
        pool = self.fc(n.new(), pool, 32)
        return pool

    def buildRegression(self, inputFace, inputEyes):
        n = NameGenerator('regFc')
        pool = tf.concat([inputFace, inputEyes], 1)
        pool = self.fc(n.new(), pool, 64)
        pool = self.reg(n.new(), pool, 2, opName = "output")
        return pool

    def buildLoss(self, output, label, useLearningRateDecay = True):
        with tf.name_scope('loss'):
            self.error = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(output - label), axis = 1)))
            tf.summary.scalar('error', self.error)
            self.loss = tf.square(self.error)
            tf.summary.scalar('loss', self.loss)
            self.weightDecayLoss = nn.weightDecayLoss(self.loss)
            tf.summary.scalar('weightDecayLoss', self.weightDecayLoss)
            self.errorDegree = tf.atan(self.error) / 3.141592 * 180
            tf.summary.scalar('errorDegree', self.errorDegree)
            self.errorCm = self.error * 40.0
            tf.summary.scalar('errorCm', self.errorCm)
        return self.loss

    def buildTrainer(self, loss, learningRate = 0.001):
        self.global_step = tf.Variable(0, trainable=False)
        if(self.useRateDecay):
            decay_r = self.dataSize / self.batchSize * self.rateDecayEpoch
            self.learning_rate = tf.train.exponential_decay(learningRate, self.global_step, int(decay_r), 0.7, staircase=True)
        else:
            self.learning_rate = tf.constant(learningRate)
        tf.summary.scalar('learningRate', self.learning_rate)
        self.adamOptimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.sgdOptimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate * 10)
        self.adamTrainStep = nn.gradientClippedMinimize(self.adamOptimizer, loss, global_step=self.global_step, useClip = True)
        if self.useWeightDecay:
            loss = self.weightDecayLoss
        self.sgdTrainStep = nn.gradientClippedMinimize(self.sgdOptimizer, loss, global_step=self.global_step, useClip = True)
    
    def getLoss(self):
        if(self.getTrainStep() == self.sgdOptimizer and self.useWeightDecay):
            return self.weightDecayLoss
        return self.loss

    def getTrainStep(self):
        if(self.ephoc > self.rateDecayEpoch and self.useSwitching):
            return self.sgdTrainStep
        return self.adamTrainStep

    def getFeedDict(self, batchData, isTrain):
        feed = \
        { 
            self.inputLeft : batchData.left, 
            self.inputRight : batchData.right, 
            self.inputFace : batchData.face, 
            self.inputLabel : batchData.label, 
            self.keep_prob : self.dropRate, 
            self.phase_train : True 
        }
        if not isTrain:
            feed[self.phase_train] = False
            feed[self.keep_prob] = self.testDropRate
        return feed

    def forward(self, sess, batchData, summary = None):
        feed = self.getFeedDict(batchData, False)
        fetch = [ self.getLoss(), self.error ]
        if not summary is None:
            fetch.append(summary)
        result = sess.run(fetch, feed_dict = feed)
        del fetch[:]
        del feed, fetch
        if not summary is None:
            return result[0 : len(result)-1], result[-1]
        return result

    def optimize(self, sess, batchData, summary = None):
        self.step += 1
        self.ephoc = int(self.step * self.batchSize / self.dataSize)
        feed = self.getFeedDict(batchData, True)
        fetch = [ self.getTrainStep(), self.getLoss(), self.error ]
        if not summary is None:
            fetch.append(summary)
        result = sess.run(fetch, feed_dict = feed)
        del fetch[:]
        del feed, fetch
        if not summary is None:
            return result[1 : len(result) - 1], result[-1]
        return result[1:]

def backup(path):
    import shutil
    import datetime
    fileMe = os.path.abspath(__file__)
    fileDist = os.path.join(path, os.path.splitext(os.path.basename(fileMe))[0] + " ["+datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H-%M-%S')+"].py")
    print("copy me to", fileDist)
    shutil.copy2(fileMe, fileDist)

def loadData(eyeSize, faceSize):
    basedir = "C:\\Library\\koi 2017\\Source\\GazeDataset\\"
    dataList = [
        basedir + "eyesub1\\", 
        basedir + "eyesub2\\", 
        basedir + "eyesub3\\", 
        basedir + "eyesub4\\", 
        basedir + "eyesub5\\", 
        basedir + "eyesub6\\", 
        basedir + "eyesub7\\", 
        basedir + "eyesub8\\", 
        basedir + "eyesub9\\", 
        basedir + "eyesub10\\", 
        basedir + "eyesub11\\", 
        basedir + "eyesub12\\", 
        basedir + "eyesub13\\", 
        basedir + "eyesub14\\", 
        basedir + "eyesub15\\", 
        basedir + "eyesub16\\", 
        basedir + "eyesub17\\", 
        basedir + "eyesub18\\", 
        basedir + "eyesub19\\", 
        basedir + "eyesub20\\", 
        basedir + "eyesub21\\", 
        basedir + "eyesub22\\", 
        basedir + "eyesub23\\", 
        basedir + "eyesub24\\",
        basedir + "eyesub25\\",
        ]
    data = Dataset(dataList, eyeSize = eyeSize, faceSize = faceSize)
    return data

class ModelTester:
    def plot(self):
        length = self.label.shape[0]
        plt.ylim(-1,1)
        plt.xlim(-1,1)
        #errFac = 1/max(errors)
        for i in range(length):
            diff = self.result[i] - self.label[i]
            #error = np.sqrt(np.sum(np.square(diff)))
            plt.arrow(self.result[i][0], self.label[i][1], -diff[0], -diff[1], head_width=0.013, width=0.003, color=matplotlib.colors.to_rgba((1,0,0,1-max(0,0))))
        plt.show()
    def test(self, model, batchData, useBnorm = False, useSELU = True):
        inputLeft = model.inputLeft
        inputRight = model.inputRight
        inputFace = model.inputFace
        phase_train = model.phase_train
        keep_prob = model.keep_prob
        output = model.output
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True        
        dropRate = 1.0
        if useSELU:
            dropRate = 0.0
        sess = model.sess
        fetch = sess.run([output], { inputLeft : batchData.left, inputRight : batchData.right, inputFace : batchData.face, phase_train : False, keep_prob : dropRate })
        result = np.sqrt(np.average(np.square(fetch[0] - batchData.label)) * 2)
        self.label =  batchData.label
        self.result = fetch[0]
        print(result)
        return result

class ModelLoader:
    def __init__(self, pbFile, useBnorm = False, useSELU = True):
        self.graph = self.load_graph(pbFile)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config, graph = self.graph)
        self.inputLeft = self.graph.get_tensor_by_name('name/input_left:0')
        self.inputRight = self.graph.get_tensor_by_name('name/input_right:0')
        self.inputFace = self.graph.get_tensor_by_name('name/input_face:0')
        self.keep_prob = self.graph.get_tensor_by_name('name/keep_prob:0')
        self.output = self.graph.get_tensor_by_name('name/output:0')
        if(useBnorm or useSELU):
            self.phase_train = self.graph.get_tensor_by_name('name/phase_train:0')
    
    def load_graph(self, frozen_graph_filename):
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def, 
                input_map=None, 
                return_elements=None, 
                name="name", 
                op_dict=None, 
                producer_op_list=None
            )
        return graph

class ModelSaver:
    def __init__(self, parentPath, checkpointName, useBnorm = False):
        self.parentPath = parentPath
        self.checkpointName = checkpointName
        self.useBnorm = useBnorm
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
        self.inputLeft = graph.get_tensor_by_name('input_left:0')
        self.inputRight = graph.get_tensor_by_name('input_right:0')
        self.inputFace = graph.get_tensor_by_name('input_face:0')
        self.keep_prob = graph.get_tensor_by_name('keep_prob:0')
        self.output = graph.get_tensor_by_name('output:0')
        self.phase_train = graph.get_tensor_by_name('phase_train:0')
    
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
        gd = graph_util.convert_variables_to_constants(self.sess, gd, ["output"])

        optlib.ensure_graph_is_valid(gd)
        input_node_names = ["input_left", "input_right", "input_face", "keep_prob", "phase_train"]
        output_node_names = ["output"]
        placeholder_type_enum = [tf.float32, tf.float32, tf.float32, tf.float32, tf.bool]
        for i in range(len(placeholder_type_enum)):
            placeholder_type_enum[i] = placeholder_type_enum[i].as_datatype_enum
        print("strip...")
        gd = strip_unused_lib.strip_unused(gd, input_node_names, output_node_names, placeholder_type_enum)
        optlib.ensure_graph_is_valid(gd)
        filename = 'frozen ' + time.strftime(R" %m-%d_%H-%M-%S", time.localtime()) + '.pb'
        tf.train.write_graph(gd, self.parentPath, filename, as_text=False)
        return os.path.join(self.parentPath, filename)

def testAndFreeze(sessName):
    def listDirs(dir):
        for (_, dirs, files) in os.walk(dir):
            return dirs
    def listFiles(dir):
        for _, dirs, files in os.walk(dir):
            return files
    data = loadData(eyeSize = 60, faceSize = 32)
    targetDir = './temp/' + sessName
    files = listFiles(targetDir)
    ckptNames = []
    for f in files:
        if f.endswith('.meta'):
            cname = os.path.basename(f)[:-5]
            ckptNames.append(cname)
    ckptNames.sort()
    model = ModelSaver(targetDir, ckptNames[-1])
    pbFile = model.freeze()
    frozen = ModelLoader(pbFile)
    tester = ModelTester()
    tester.test(frozen, data.batchTest(100), useBnorm = False, useSELU = True)
    tester.plot()
    tester.test(model, data.batchTest(100), useBnorm = False, useSELU = True)
    tester.plot()

def train():
    faceSize = 32
    eyeSize = 60
    batchSize = 50
    step = 0
    lastEphoc = 0
    fpsCounter = FpsCounter()

    data = loadData(faceSize = faceSize, eyeSize = eyeSize)
    model = Model(\
        faceSize = faceSize, 
        eyeSize = eyeSize, 
        dataSize = data.count,
        batchSize = batchSize, 
        useSELU = True,
        useRateDecay = True,
        rateDecayEpoch = 12,
        useSwitching = True,
        useWeightDecay = True,
        useMobileNet = True)
    nn.weightReport()

    def signal_handler(signal, frame):
        print("Program EXIT ==================")
        data.close()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        merged = tf.summary.merge_all()
        timestemp = time.strftime(R" %m-%d_%H-%M-%S", time.localtime())
        testDirName = './temp/test' + timestemp
        trainWriter = tf.summary.FileWriter('./temp/train' + timestemp, sess.graph, filename_suffix='train', flush_secs=20)
        testWriter = tf.summary.FileWriter(testDirName, sess.graph, filename_suffix='test', flush_secs=20)
        backup(testDirName)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        while True:
            #update
            ephoc = int(float(step) * batchSize / data.count)
            step += 1
            fpsCounter.add(batchSize)
            #train
            batch = data.batch(batchSize)
            fetch, summary = model.optimize(sess, batch, summary = merged)
            trainWriter.add_summary(summary, step)
            #test
            if(step % 10 == 0):
                testBatch = data.batchTest(batchSize)
                tfetch, tsummary = model.forward(sess, testBatch, summary = merged)
                testWriter.add_summary(tsummary, step)
                print \
                ( \
                    'step:', step, 
                    'epoch:', ephoc, '(%0.2f%%)' % (float(step * batchSize % data.count) / data.count * 100.0), 
                    'fetch:', fetch, 
                    'tfetch:', tfetch, 
                    'data/s:', fpsCounter.fps()
                )
                testBatch.dispose()
                del tfetch[:]
                del testBatch, tfetch, tsummary
            #new ephoc
            if(lastEphoc != ephoc):
                ckpt_path = saver.save(sess, testDirName + '/model.ckpt', global_step=step)
                print("checkpoint saved : ", ckpt_path)
                gc.collect()
                lastEphoc = ephoc
                del ckpt_path
            #dispose
            batch.dispose()
            del fetch[:]
            del fetch, summary, batch

if (__name__ == '__main__'):
    #train()
    testAndFreeze('test 04-07_18-54-48')
