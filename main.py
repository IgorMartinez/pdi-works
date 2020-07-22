import numpy as np
import math as math
import extraction
import svm
import dtree
import multilayer_perceptron as mlpt
import commons as cms
import os
from sklearn.model_selection import train_test_split

CHARACTERISTICS_CSV = 'characteristics.csv'
CLASSIFICATION_CSV = 'classification.csv'
METHODS = [svm, dtree]
SEED = 24

def main(ofile, sep=';'):
    y = np.array(['benigno']*20 + ['maligno']*20)
    db = cms.buildDatabase()
    header = ['id', 'channel', 'test_size'] + [m.NAME for m in METHODS]
    table = []
    xp = 1
    if os.path.exists(CHARACTERISTICS_CSV):
        table = cms.loadCaracteristics(CHARACTERISTICS_CSV)
    else:
        table = extraction.extract(db)
        cms.saveCaracteristics(table, CHARACTERISTICS_CSV)
    with open(ofile, "w") as f:
        f.write(sep.join(header) + '\n')
        for ch in ['red', 'green', 'blue', 'gray']:
            for method in METHODS:
                method.accur = []
            for size in np.arange(0.1, 1.0, 0.1):
                for test in range(500):
                    print('[DBG] %.2f' % size)
                    while True:
                        xs, xt, ys, yt = train_test_split(table[ch], y, test_size=size, random_state=None)
                        if len(np.unique(ys)) > 1:
                            break
                    for method in METHODS:
                        method.accur.append(method.classify(xs, xt, ys, yt))
                f.write(sep.join([str(xp), ch, str(size)] + [str(np.mean(m.accur)) for m in METHODS]) + '\n')
                xp += 1

if __name__ == "__main__":
    main(CLASSIFICATION_CSV)

                

    