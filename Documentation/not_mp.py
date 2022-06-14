
from multiprocessing import Pool, Process
import numpy as np
import time


class mp:

    def __init__(self):
        pass

    def compute_eibv(self, ind):
        return ind / 2

    def test_mp(self):
        indc = np.arange(100)
        t1 = time.time()
        p = []
        for i in indc:
            tp = Process(target=self.compute_eibv, args=[i])
            tp.start()
            p.append(tp)
        t2 = time.time()
        print(t2 - t1)
        t1 = time.time()
        for i in indc:
            self.compute_eibv(i)
        t2 = time.time()
        print(t2 - t1)
        for pp in p:
            pp.join()

        pass

if __name__ == "__main__":
    m = mp()
    m.test_mp()
