"""Closed-form reference solutions used to verify the FreeFEM solvers and the PINN.

All quantities in SI units unless stated otherwise.
"""
import numpy as np
from scipy.special import erfc, erfcx

MU0 = 4e-7 * np.pi


def diffusion_dirichlet(x, t, D, c0, cs):
    """Semi-infinite body, constant D, surface held at cs (Crank, The Mathematics of Diffusion, ch. 3)."""
    return c0 + (cs - c0) * erfc(x / (2.0 * np.sqrt(D * t)))


def diffusion_robin(x, t, D, beta, c0, cp):
    """Semi-infinite body, constant D, surface flux -D dc/dx = beta (cp - c).

    Crank, The Mathematics of Diffusion, ch. 3; Carslaw & Jaeger, Conduction of Heat in Solids, ch. 2:
      (c - c0)/(cp - c0) = erfc(z) - exp(h x + h^2 D t) erfc(z + h sqrt(D t)),
      z = x / (2 sqrt(D t)), h = beta / D.
    The second term is evaluated as exp(-z^2) erfcx(z + h sqrt(Dt)) to avoid overflow.
    Also covers convective quenching with D -> alpha and beta -> h_conv / (rho cp).
    """
    z = x / (2.0 * np.sqrt(D * t))
    u = z + (beta / D) * np.sqrt(D * t)
    return c0 + (cp - c0) * (erfc(z) - np.exp(-z**2) * erfcx(u))


def skin_depth(freq, sigma, mur=1.0):
    return np.sqrt(2.0 / (2 * np.pi * freq * MU0 * mur * sigma))


def eddy_current_slab(x, L, freq, sigma, mur=1.0, A0=1.0):
    """A_z in a slab with A(0)=A0, A(L)=0: A = A0 sinh(k(L-x))/sinh(kL), k=(1+i)/delta."""
    k = (1 + 1j) / skin_depth(freq, sigma, mur)
    A = A0 * np.sinh(k * (L - x)) / np.sinh(k * L)
    p = 0.5 * sigma * (2 * np.pi * freq) ** 2 * np.abs(A) ** 2
    return A, p
