import numpy as np


class Logspace:
    def __init__(self):
        self.LOGZERO = np.nan

    def logspacematprod(self, a, b):
        a = np.mat(a, dtype='float64')
        b = np.mat(b, dtype='float64')
        rows = np.shape(a)[0]
        col = np.shape(b)[1]
        if (np.shape(a)[1] != np.shape(b)[0]):
            print "error:matrix dimension didn't connects!\n"
            print a, b
            return np.nan
        c = np.zeros((rows, col))
        for i in range(rows):
            for j in range(col):
                tmp = a[i, :] + b[:, j].T
                max = tmp.max()
                tmp = tmp - max
                c[i][j] = np.log(np.sum(np.exp(tmp))) + max
        return c

    def logspacematdotprod(self, mat_a, scale_b):
        a = np.mat(mat_a, dtype='float64')
        b = np.mat(scale_b, dtype='float64')
        if (np.shape(b)[0] == 1 and np.shape(b)[1] == 1):
            return a + b

    def logsapceProduct_Max(self, rowVector_a , mat_b):
        a = np.mat(rowVector_a, dtype = 'floadt64')
        b = np.mat(mat_b,dtype = 'float64')
        col = np.shape(b)[1]
        c = np.zeros((1,col))
        d = np.zeros((1,col))

        if (np.shape(a)[1] != np.shape(b)[0]):
            print "error:matrix dimension didn't connects!\n"
            print a, b
            return np.nan
        for j  in range(col):
            tmp = a[0,:] + b[:,j].T
            max = tmp.max()
            max_index = tmp.index(max)
            c[0,j] = max
            d[0,j] = max_index
        return c,d

