import numpy as np
import math as math
import extraction
import svm
import dtree
import commons as cms
import os
from sklearn.model_selection import train_test_split

CHARACTERISTICS_CSV = 'characteristics.csv'
SEED = 'pdi'

def main(ofile, sep=';'):
    y = np.array(['benigno']*20 + ['maligno']*20)
    db = cms.buildDatabase()
    header = ['id', 'channel', 'svm_accur', 'dtree_acur', 'nn_accur']
    table = []
    if os.path.exists(CHARACTERISTICS_CSV):
        table = cms.loadCaracteristics(CHARACTERISTICS_CSV)
    else:
        table = extraction.extract(db)
        cms.saveCaracteristics(table, CHARACTERISTICS_CSV)
    with open(ofile, "w") as f:
        f.write(sep.join(header) + '\n')
        for ch in ['red', 'green', 'blue', 'gray']:
            for size in range(0.1, 1.0, 0.1):
                for method in [svm, dtree]:
                    xs, xt, ys, yt = train_test_split(table[ch], y, test_size=size, random_state=SEED)
                    accur = method.classify(xs, xt, ys, yt)

if __name__ == "__main__":
    main('classification.csv')

                

    