import numpy as np
from numpy.polynomial import Polynomial


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def tanh_prime(x: np.ndarray) -> np.ndarray:
    return 1 - np.tanh(x) ** 2


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x: np.ndarray) -> np.ndarray:
    return sigmoid(x) * (1 - sigmoid(x))


p_coeffs = np.ones(4)
p_coeffs[0] = 0
p = Polynomial(p_coeffs)
p_prime = p.deriv(m=1)
print('P:', p, 'P\':', p_prime)


def tanh_poly(x: np.ndarray) -> np.ndarray:
    global p, p_prime
    return np.tanh(p(x))


def tanh_poly_prime(x: np.ndarray) -> np.ndarray:
    global p, p_prime
    return (p_prime(x)) / np.cosh(p(x)) ** 2


def sigmoid_poly(x: np.ndarray) -> np.ndarray:
    global p, p_prime
    return 1 / (1 + np.exp(-p(x)))


def sigmoid_poly_prime(x: np.ndarray) -> np.ndarray:
    global p, p_prime
    return np.exp(p(x)) * p_prime(x) / (np.exp(p(x)) + 1) ** 2
