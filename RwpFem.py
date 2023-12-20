import sys
import numpy as np
import scipy as sp
import Quadratur as Q


def LinElem(x0, x1, k, r, q, f, intyp):
    hi = x1 - x0

    def K_func(xi):
        return [[k(F(xi))/hi**2 * dphi_lin1(xi) * dphi_lin1(xi)
                 + r(F(xi))/hi * dphi_lin1(xi) * phi_lin1(xi)
                 + q(F(xi))*phi_lin1(xi)*phi_lin1(xi),
                 k(F(xi))/hi**2 * dphi_lin2(xi) * dphi_lin1(xi)
                 + r(F(xi))/hi * dphi_lin2(xi) * phi_lin1(xi)
                 + q(F(xi))*phi_lin2(xi)*phi_lin1(xi)],
                [k(F(xi))/hi**2 * dphi_lin1(xi) * dphi_lin2(xi)
                 + r(F(xi))/hi * dphi_lin1(xi) * phi_lin2(xi)
                 + q(F(xi))*phi_lin1(xi)*phi_lin2(xi),
                 k(F(xi))/hi**2 * dphi_lin2(xi) * dphi_lin2(xi)
                 + r(F(xi))/hi * dphi_lin2(xi) * phi_lin2(xi)
                 + q(F(xi))*phi_lin2(xi)*phi_lin2(xi)]]

    def f_func(xi):
        return [f(F(xi)) * phi_lin1(xi),
                f(F(xi)) * phi_lin2(xi)]

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
        return [[k(F(xi))/hi**2 * dphi_quad1(xi) * dphi_quad1(xi)
                 + r(F(xi))/hi * dphi_quad1(xi) * phi_quad1(xi)
                 + q(F(xi)) * phi_quad1(xi) * phi_quad1(xi),
                 k(F(xi))/hi**2 * dphi_quad2(xi) * dphi_quad1(xi)
                 + r(F(xi))/hi * dphi_quad2(xi) * phi_quad1(xi)
                 + q(F(xi)) * phi_quad2(xi) * phi_quad1(xi),
                 k(F(xi))/hi**2 * dphi_quad3(xi) * dphi_quad1(xi)
                 + r(F(xi))/hi * dphi_quad3(xi) * phi_quad1(xi)
                 + q(F(xi)) * phi_quad3(xi) * phi_quad1(xi)],
                [k(F(xi))/hi**2 * dphi_quad1(xi) * dphi_quad2(xi)
                 + r(F(xi))/hi * dphi_quad1(xi) * phi_quad2(xi)
                 + q(F(xi)) * phi_quad1(xi) * phi_quad2(xi),
                 k(F(xi))/hi**2 * dphi_quad2(xi) * dphi_quad2(xi)
                 + r(F(xi))/hi * dphi_quad2(xi) * phi_quad2(xi)
                 + q(F(xi)) * phi_quad2(xi) * phi_quad2(xi),
                 k(F(xi))/hi**2 * dphi_quad3(xi) * dphi_quad2(xi)
                 + r(F(xi))/hi * dphi_quad3(xi) * phi_quad2(xi)
                 + q(F(xi)) * phi_quad3(xi) * phi_quad2(xi)],
                [k(F(xi))/hi**2 * dphi_quad1(xi) * dphi_quad3(xi)
                 + r(F(xi))/hi * dphi_quad1(xi) * phi_quad3(xi)
                 + q(F(xi)) * phi_quad1(xi) * phi_quad3(xi),
                 k(F(xi))/hi**2 * dphi_quad2(xi) * dphi_quad3(xi)
                 + r(F(xi))/hi * dphi_quad2(xi) * phi_quad3(xi)
                 + q(F(xi)) * phi_quad2(xi) * phi_quad3(xi),
                 k(F(xi))/hi**2 * dphi_quad3(xi) * dphi_quad3(xi)
                 + r(F(xi))/hi * dphi_quad3(xi) * phi_quad3(xi)
                 + q(F(xi)) * phi_quad3(xi) * phi_quad3(xi)]]

    def f_func(xi):
        return [f(F(xi)) * phi_quad1(xi),
                f(F(xi)) * phi_quad2(xi),
                f(F(xi)) * phi_quad3(xi)]

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
            Ki, fi = LinElem(xKno[i], xKno[i+1], k, r, q, f, intyp)
        else:
            Ki, fi = QuadElem(xKno[i], xKno[i+1], k, r, q, f, intyp)
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
