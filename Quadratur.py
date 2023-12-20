import numpy as np

knoten = np.array([[1/2],
                   [(1-1/np.sqrt(3))/2, (1+1/np.sqrt(3))/2],
                   [(1-np.sqrt(3/5))/2, 1/2, (1+np.sqrt(3/5))/2]],
                  dtype=object)

gewichte = np.array([[1],
                     [1/2, 1/2],
                     [5/18, 4/9, 5/18]],
                    dtype=object)


def gauss(f, intyp):
    x = knoten[intyp - 1]
    w = gewichte[intyp - 1]
    res = 0
    for i in range(len(x)):
        res = res + w[i]*f(x[i])
    return res
