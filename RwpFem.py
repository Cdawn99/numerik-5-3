import sys
import numpy as np
import scipy as sp
import Quadratur as Q


def LinElem(x0, x1, k, r, q, f, intyp):
    hi = x1 - x0

    def K_func(xi):
        return np.array(
                [[k(F(xi, x0, x1))/hi**2 * dphi_lin1(xi) * dphi_lin1(xi)
                 + r(F(xi, x0, x1))/hi * dphi_lin1(xi) * phi_lin1(xi)
                 + q(F(xi))*phi_lin1(xi)*phi_lin1(xi),
                 k(F(xi, x0, x1))/hi**2 * dphi_lin2(xi) * dphi_lin1(xi)
                 + r(F(xi, x0, x1))/hi * dphi_lin2(xi) * phi_lin1(xi)
                 + q(F(xi, x0, x1))*phi_lin2(xi)*phi_lin1(xi)],
                 [k(F(xi, x0, x1))/hi**2 * dphi_lin1(xi) * dphi_lin2(xi)
                 + r(F(xi, x0, x1))/hi * dphi_lin1(xi) * phi_lin2(xi)
                 + q(F(xi, x0, x1))*phi_lin1(xi)*phi_lin2(xi),
                 k(F(xi, x0, x1))/hi**2 * dphi_lin2(xi) * dphi_lin2(xi)
                 + r(F(xi, x0, x1))/hi * dphi_lin2(xi) * phi_lin2(xi)
                 + q(F(xi, x0, x1))*phi_lin2(xi)*phi_lin2(xi)]]
                 )

    def f_func(xi):
        return np.array(
                [f(F(xi, x0, x1)) * phi_lin1(xi),
                 f(F(xi, x0, x1)) * phi_lin2(xi)]
                )

    if intyp == 0:
        Ki = hi * sp.integrate.quad(K_func, 0, 1)
        fi = hi * sp.integrate.quad(f_func, 0, 1)
    else:
        Ki = hi * Q.gauss(K_func, intyp)
        fi = hi * Q.gauss(f_func, intyp)
    return Ki, fi


def QuadElem(x0, x1, k, r, q, f, intyp):
    hi = x1 - x0

    def K_func(xi):
        return np.array(
                [[k(F(xi, x0, x1))/hi**2 * dphi_quad1(xi) * dphi_quad1(xi)
                 + r(F(xi, x0, x1))/hi * dphi_quad1(xi) * phi_quad1(xi)
                 + q(F(xi, x0, x1)) * phi_quad1(xi) * phi_quad1(xi),
                 k(F(xi, x0, x1))/hi**2 * dphi_quad2(xi) * dphi_quad1(xi)
                 + r(F(xi, x0, x1))/hi * dphi_quad2(xi) * phi_quad1(xi)
                 + q(F(xi, x0, x1)) * phi_quad2(xi) * phi_quad1(xi),
                 k(F(xi, x0, x1))/hi**2 * dphi_quad3(xi) * dphi_quad1(xi)
                 + r(F(xi, x0, x1))/hi * dphi_quad3(xi) * phi_quad1(xi)
                 + q(F(xi, x0, x1)) * phi_quad3(xi) * phi_quad1(xi)],
                 [k(F(xi, x0, x1))/hi**2 * dphi_quad1(xi) * dphi_quad2(xi)
                 + r(F(xi, x0, x1))/hi * dphi_quad1(xi) * phi_quad2(xi)
                 + q(F(xi, x0, x1)) * phi_quad1(xi) * phi_quad2(xi),
                 k(F(xi, x0, x1))/hi**2 * dphi_quad2(xi) * dphi_quad2(xi)
                 + r(F(xi, x0, x1))/hi * dphi_quad2(xi) * phi_quad2(xi)
                 + q(F(xi, x0, x1)) * phi_quad2(xi) * phi_quad2(xi),
                 k(F(xi, x0, x1))/hi**2 * dphi_quad3(xi) * dphi_quad2(xi)
                 + r(F(xi, x0, x1))/hi * dphi_quad3(xi) * phi_quad2(xi)
                 + q(F(xi, x0, x1)) * phi_quad3(xi) * phi_quad2(xi)],
                 [k(F(xi, x0, x1))/hi**2 * dphi_quad1(xi) * dphi_quad3(xi)
                 + r(F(xi, x0, x1))/hi * dphi_quad1(xi) * phi_quad3(xi)
                 + q(F(xi, x0, x1)) * phi_quad1(xi) * phi_quad3(xi),
                 k(F(xi, x0, x1))/hi**2 * dphi_quad2(xi) * dphi_quad3(xi)
                 + r(F(xi, x0, x1))/hi * dphi_quad2(xi) * phi_quad3(xi)
                 + q(F(xi, x0, x1)) * phi_quad2(xi) * phi_quad3(xi),
                 k(F(xi, x0, x1))/hi**2 * dphi_quad3(xi) * dphi_quad3(xi)
                 + r(F(xi, x0, x1))/hi * dphi_quad3(xi) * phi_quad3(xi)
                 + q(F(xi, x0, x1)) * phi_quad3(xi) * phi_quad3(xi)]]
                 )

    def f_func(xi):
        return np.array(
                [f(F(xi, x0, x1)) * phi_quad1(xi),
                 f(F(xi, x0, x1)) * phi_quad2(xi),
                 f(F(xi, x0, x1)) * phi_quad3(xi)]
                )

    if intyp == 0:
        Ki = hi * sp.integrate.quad(K_func, 0, 1)
        fi = hi * sp.integrate.quad(f_func, 0, 1)
    else:
        Ki = hi * Q.gauss(K_func, intyp)
        fi = hi * Q.gauss(f_func, intyp)
    return Ki, fi


def Fehlersh():
    return


def RwpFem1d(xGit, k, r, q, f, rba, rbb, eltyp, intyp):
    if eltyp != 1 and eltyp != 2:
        sys.exit(1)
    Me = len(xGit) - 1
    Ng = eltyp*Me
    xKno = np.linspace(xGit[0], xGit[1], Ng+1)
    KnEl = KnElGen(Me, eltyp, intyp)
    Kh = sp.sparse.csc_matrix((Ng+1, Ng+1))
    fh = np.zeros(Ng+1)
    for i in range(0, Me):
        if eltyp == 1:
            Ki, fi = LinElem(xKno[i], xKno[i+1], k, r, q, f, intyp)
        else:
            Ki, fi = QuadElem(xKno[2*i], xKno[2*i+2], k, r, q, f, intyp)
        for j in range(1, eltyp+2):
            v = int(KnEl[i, j+1])
            fh[v] = fh[v] + fi[j-1]
            for l in range(1, eltyp+2):
                s = int(KnEl[i, l+1])
                Kh[v, s] = Kh[v, s] + Ki[j-1, l-1]
    typa, kapa, mua = rba
    typb, kapb, mub = rbb
    if typa == 1:
        Kh[0] = 0
        fh[0] = 0
        fh = fh - np.matmul(Kh.toarray(), np.array([mua] + [0]*(Ng)))
        Kh[0, 0] = 1
        fh[0] = mua
    elif typa == 2:
        fh[0] = fh[0] + mua
    elif typa == 3:
        Kh[0, 0] = Kh[0, 0] + kapa
        fh[0] = fh[0] + mua
    if typb == 1:
        Kh[Ng - 1] = 0
        fh[Ng - 1] = 0
        fh = fh - np.matmul(Kh.toarray(), np.array([0]*(Ng) + [mub]))
        Kh[Ng - 1, Ng - 1] = 1
        fh[Ng - 1] = mub
    elif typb == 2:
        fh[Ng - 1] = fh[Ng - 1] + mub
    elif typb == 3:
        Kh[Ng - 1, Ng - 1] = Kh[Ng - 1, Ng - 1] + kapb
        fh[Ng - 1] = fh[Ng - 1] + mub
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


def F(xi, x0, x1): return (x1 - x0)*xi + x0


def phi_lin1(xi): return 1 - xi
def phi_lin2(xi): return xi
def phi_quad1(xi): return (2*xi - 1)*(xi - 1)
def phi_quad2(xi): return 4*xi*(1 - xi)
def phi_quad3(xi): return xi*(2*xi - 1)


def dphi_lin1(xi): return -1
def dphi_lin2(xi): return 1
def dphi_quad1(xi): return 4*xi - 3
def dphi_quad2(xi): return 4 - 8*xi
def dphi_quad3(xi): return 4*xi - 1
