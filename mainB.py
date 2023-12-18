import numpy as np
import matplotlib. pyplot as plt

import RwpFem as FEM

# Gitter
a, b = 0, 1
Nnod = 10+1
xGit = np.linspace(a, b, Nnod)


# Funktionen
def exa(x): return np.exp(x) + np.exp(-x) + 1  # exakte Loesung
def kco(x): return 1 + x
def qco(x): return 1 + x
def rco(x): return 1 + x
def fco(x): return x * (np.exp(x) - np.exp(-x) + 1) + 1


# Randbedingungen
# Dirichlet - RB links
rba = np.array([1, 0, exa(a)])
# Robin - RB rechts
rbb = np.array([3, 1, 3*np.exp(1) - np.exp(-1) + 1])

# Auswahl Ansatz linear bzw. quadratisch und der Quadraturformel
eltyp, intyp = 2, 3
# Assemblierung und Loesung
uw, xw = FEM.RwpFem1d(xGit, kco, rco, qco, fco, rba, rbb, eltyp, intyp)
# Vergleich mit exakter Loesung in Knoten
uexa = exa(xw)
print(f"Ansatz: {eltyp}, QuadrTyp: {intyp}, Max.Fehler: {max(abs(uw - uexa))}")

# Einfache Visualisierung
fig = plt.figure()
plt.plot(xw, uw, 'c', marker='P', label='Numerische Loesung')
plt.plot(xw, uexa, 'm', label='Exakte Loesung')
plt.grid(visible=True)
plt.title("FEM - Bsp b")
plt.xlabel("Ort x")
plt.ylabel("Loesung u")
plt.legend()
plt.show()
