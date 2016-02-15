import numpy as np

a = np.random.randn(9, 6)

U, s, V = np.linalg.svd(a, full_matrices=True)
print U.shape, V.shape, s.shape
S = np.zeros((9, 6), dtype=complex)
S[:6, :6] = np.diag(s)


print s

print V
print type(V)
print V[1]

A = np.array([[1, 2], [3, 4]])
print A.shape

test = A * A
test2 = np.dot(A, A)

x1 = np.arange(9.0).reshape((3, 3))
x2 = np.arange(3.0)
print np.multiply(x1, x2)
print x1 * x2

print x1
print x2
