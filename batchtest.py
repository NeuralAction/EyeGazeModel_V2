from main import *
import concurrent.futures
import time

def main():
    data = loadData(eyeSize = 64, faceSize = 64)
    fps = FpsCounter()
    step = 0
    while True:
        step += 1
        batch = data.batch(100)
        time.sleep(1)
        fps.add(100)
        if(step % 10 == 0):
            print('data/sec', fps.fps())
        batch.dispose()
        del batch

if __name__ == '__main__':
    main()
