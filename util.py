import codecs
import math
import os
import random
import sys
import time
from multiprocessing import Pool
import multiprocessing

import numpy as np
import concurrent.futures

def getTimeStamp():
    timestemp = time.strftime(R"%m-%d_%H-%M-%S", time.localtime())
    return timestemp

# def backupSource(path):
#     import shutil
#     fileMe = os.path.abspath(__file__)
#     fileDist = os.path.join(path, os.path.splitext(os.path.basename(fileMe))[0] + " [" + getTimeStamp() + "].py")
#     print("copy me to", fileDist)
#     shutil.copy2(fileMe, fileDist)

class ThreadBuffer:
    def __init__(self):
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.function = None
        self.result = None
        self.lastArgs = None
        self.lastFunction = None
    
    def checkRefresh(self, function, args):
        if self.result is None or self.lastArgs is None or self.lastFunction is None:
            return True
        if self.lastFunction is function:
            return False
        if len(self.lastArgs) is len(args):
            for i in range(len(args)):
                if not(self.lastArgs[i] is args[i]):
                    return True
            return False
        return True
    
    def queue(self, function, args):
        self.result = self.pool.submit(function, *args)
        self.lastArgs = args
        self.lastFunction = function

    def get(self, function, args):
        if self.checkRefresh(function, args):
            self.queue(function, args)
        ret = self.result.result()
        del self.result
        self.queue(function, args)
        return ret

    def close(self):
        self.pool.shutdown()

class FpsCounter:
    def __init__(self):
        self.count = 0
        self.timestart = -1

    def add(self, count=1):
        if(self.timestart == -1):
            self.timestart = time.time()
        self.count += count
        
    def fps(self):
        fps = self.count / (time.time() - self.timestart)
        self.count = 0
        self.timestart = -1
        return fps

class NameGenerator:
    def __init__(self, header = ''):
        self.header = header
        self.index = 0
    
    def new(self):
        name = self.header + str(self.index)
        self.index += 1
        return name

class Parallel:
    def __init__(self, numWorkers = None):
        if(numWorkers is None):
            numWorkers = multiprocessing.cpu_count()
        self.workerCount = numWorkers
        self.pool = Pool(processes = numWorkers)
    
    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict
    
    def close(self):
        self.pool.close()
        self.pool.join()
    
    def repeat(self, args):
        ret = []
        for i in args:
            ret.append(self.function(i))
        return ret
    
    def map(self, function, args):
        remap = []
        for i in range(len(args)):
            procId = i % self.workerCount
            if(procId >= len(remap)):
                remap.append([])
            remap[procId].append(args[i])
        self.function = function
        result = self.pool.map(self.repeat, remap)
        resultMap = []
        for r in result:
            for item in r:
                resultMap.append(item)
            del r[:]
        del result[:]
        del result
        del remap[:]
        del remap
        return resultMap