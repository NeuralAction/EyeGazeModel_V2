import os
import codecs
import random
import math
import numpy as np
import multiprocessing
import scipy
from scipy import ndimage
from scipy import misc
from skimage import transform
from skimage import util

class dataLabel:
    def __init__(self, ind, rod1, rod2, rod3):
        self.ind = ind
        self.rod1 = rod1
        self.rod2 = rod2
        self.rod3 = rod3

def decodeLabel(str):
    nameSpl = str.split(",")
    return dataLabel(int(nameSpl[0]), float(nameSpl[1]), float(nameSpl[2]), float(nameSpl[3]))

class dataModel:
    def __init__(self, width, height, mmwidth, mmheight, originX, originY, originZ, sub):
        self.width = width
        self.height = height
        self.mmwidth = mmwidth
        self.mmheight = mmheight
        self.originX = originX
        self.originY = originY
        self.originZ = originZ
        self.sub = sub

def decodeModel(file):
    fmodel = open(file, "r", -1, "utf-8")
    lines = fmodel.readlines()
    model = dataModel(-1,-1,-1,-1,-1,-1,-1,None)
    for l in lines:
        spl = l.split(':')
        head = spl[0]
        content = spl[1]
        content = content.replace("\n", "")
        print(head)
        print(content)
        if head == "scr":
            spl = content.split(',')
            print(spl)
            model.width = float(spl[0])
            model.height = float(spl[1])
        elif head == "scrmm":
            spl = content.split(',')
            print(spl)
            model.mmwidth = float(spl[0])
            model.mmheight = float(spl[1])
        elif head == "scrorigin":
            spl = content.split(',')
            print(spl)
            model.originX = float(spl[0])
            model.originY = float(spl[1])
            model.originZ = float(spl[2])
        elif head == "sub":
            model.sub = content
        else:
            print("ERROR while reading model.txt " + l)
    return model

class Processor:
    def __init__(self, randomize, randmul, randadd, normalize, imagesize, anglemul, rotate, randpad):
        self.randomize = randomize
        self.randmul = randmul
        self.randadd = randadd
        self.normalize = normalize
        self.imagesize = imagesize
        self.anglemul = anglemul
        self.rotate = rotate
        self.randpad = randpad

    def __call__(self, args):
        first = True
        returnBat = []
        returnLab = []
        for item in args:
            img_decode = misc.imread(item.filename)
            img_resize = misc.imresize(img_decode, [self.imagesize, self.imagesize])
            img = img_resize
            if(self.randomize):
                rand_mul = random.random() * self.randmul
                rand_mul = 1 - rand_mul * 0.8 + rand_mul * 0.2
                rand_add = random.random() * (self.randadd * 2) - self.randadd
                img = img * rand_mul + rand_add
                percent = self.randpad
                def randpad():
                    return int(self.imagesize * percent * np.random.random())
                img = np.lib.pad(img, ((randpad(),randpad()),(randpad(),randpad()),(0,0)), 'constant', constant_values=(0.0,))
                angle = self.rotate
                img = transform.rotate(img, np.random.random() * angle - angle / 2)
                def randpad():
                    return int(self.imagesize * percent * 0.6 * np.random.random())
                crop = randpad()
                img = util.crop(img,((randpad()+crop,randpad()+crop),(randpad()+crop,randpad()+crop),(0,0)))
                img = transform.resize(img, (self.imagesize, self.imagesize))
                img = img + np.random.random(img.shape) * (random.random() * img.std())
                np.clip(img, 0, 255, out=img)
            if(self.normalize):
                #img = img / 127.5 - 1
                #img = img / 255.0
                img = img - np.average(img)
                std = np.std(img)
                if not((abs(std) < 0.01) or math.isnan(std) or math.isinf(std)):
                    img = img / std
                else:
                    img = img / 63.5
            img = np.reshape(img, [1, self.imagesize, self.imagesize, 3])
            rod1 = item.label.rod1
            rod2 = item.label.rod2
            lb = [[ rod1, rod2 ]]
            if(first):
                returnBat = img
                returnLab = lb
                first = False
            else:
                returnBat = np.concatenate([returnBat, img], 0)
                returnLab = np.concatenate([returnLab, lb], 0)
            del img_decode, img_resize, img, lb, rod1, rod2
        return ProcRet(returnBat, returnLab)

class ProcArg:
    def __init__(self, filename, label):
        self.filename = filename
        self.label = label

class ProcRet:
    def __init__(self, bat, label):
        self.bat = bat
        self.label = label

class dataWrap:
    def __init__(self, image, label, model, pool):
        self.image = image
        self.label = label
        self.size = len(label)
        self.model = model
        self.imagesize = 160
        self.anglemul = 1
        self.randmul = 0.8
        self.pool=pool
        self.randadd = 25
        self.rotate = 120
        self.randpad = 0.3
    def batch(self, count, normalize = True, randomize = True):
        bat = []
        label = []
        files = []
        for i in range(16):
            files.append([])
        for i in range(count):
            filesInd = i % 16
            ind = random.randrange(0, len(self.label))
            files[filesInd].append(ProcArg(self.image[ind], self.label[ind]))
        proc=Processor(randomize, self.randmul, self.randadd, normalize, self.imagesize, self.anglemul, self.rotate, self.randpad)
        results=self.pool.map(proc,files)
        for r in results:
            if np.any(r.bat):
                if np.any(bat):
                    bat = np.concatenate([bat, r.bat], 0)
                    label = np.concatenate([label, r.label], 0)
                else:
                    bat = r.bat
                    label = r.label
        del results, proc, files
        return bat, label

def decodeData(parentlist, pool):
    images = []
    label = []
    model = None
    for parentpath in parentlist:
        for (path, dir, files) in os.walk(parentpath):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext == ".jpg":
                    filepath = path + filename
                    name = os.path.splitext(filename)[0]
                    images.append(filepath)
                    label.append(decodeLabel(name))
                elif ext == ".txt":
                    modelTxt = path + filename
                    model = decodeModel(modelTxt)
    print("searched: " + str(len(images)))
    print("Model READ COMP")
    return dataWrap(images, label, model, pool)

if __name__ == "__main__":
    p = multiprocessing.Pool(processes=12)
    basedir = "C:\\Library\\koi 2017\\Source\\OpenDataset\\"
    dataListOpen = [basedir+"open1\\left\\",
                    basedir+"open1\\right\\",
                    basedir+"open2\\left\\",
                    basedir+"open2\\right\\",]
    dataListClose = [basedir+"close1\\left\\",
                    basedir+"close1\\right\\",
                    basedir+"close2\\left\\",
                    basedir+"close2\\right\\",
                    basedir+"close3\\left\\",
                    basedir+"close3\\right\\",]
    data = decodeData(dataListClose+dataListOpen, p)
    data.imagesize = 64
    print("readFIN")
    bat, label = data.batch(10, randomize = True)
    print("BATCH FIN")
    print(bat)
    print(label)
    print(np.std(bat))
    print(np.std(label))
    print(bat.shape)
    print(label.shape)
    show = bat[0]
    show = (show - show.min())
    show = show * (255.0 / show.max())
    import image
    image.imshow(show.astype(np.uint8))