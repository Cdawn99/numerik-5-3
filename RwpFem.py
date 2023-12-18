import sys
import numpy as np
import scipy as sp


def LinElem(x0, x1, k, r, q, f, intyp):
    return


def QuadElem(x0, x1, k, r, q, f, intyp):
    return


def Fehlersh():
    return


def RwpFem1d(xGit, k, r, q, f, rba, rbb, eltyp, intyp):
    if eltyp != 1 or eltyp != 2:
        sys.exit(1)
    Me = len(xGit) - 1
    Ng = eltyp*Me
    xKno = np.linspace(xGit[0], xGit[1], Ng)
    KnEl = KnElGen(Me, eltyp, intyp)
    Kh = sp.sparse.csc_matrix((Ng, Ng))
    fh = np.zeros(Ng)
    for i in range(1, Me+1):
        if eltyp == 1:
            Ki, fi = LinElem()
        else:
            Ki, fi = QuadElem()
        for j in range(1, eltyp+2):
            r = KnEl[i, j+1]
            fh[r] = fh[r] + fi[j]
            for l in range(1, eltyp+2):
                s = KnEl[i, l+1]
                Kh[r, s] = Kh[r, s] + Ki[j, l]
    # TODO: Randbedingungen
    uKno = sp.sparse.linalg.spsolve(Kh, fh)
    return uKno, xKno


def KnElGen(Me, eltyp, intyp):
    if eltyp == 1:
        KnEl = np.zeros((Me, 4))
    else:
        KnEl = np.zeros((Me, 5))
    KnEl[:, 0] = eltyp
    KnEl[:, 1] = intyp
    for i in range(Me):
        if eltyp == 1:
            KnEl[i, 2:] = [i, i+1]
        else:
            KnEl[i, 2:] = [2*i, 2*i+1, 2*i+2]
    return KnEl
