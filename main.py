import numpy as np
import matplotlib.pyplot as plt

#A[1600 x 1600]
A = np.loadtxt('A.dat', delimiter=',')

#xt[1600 x 1]
xt = np.ones((A.shape[1], 1))

#b[1600 x 1]
b = np.dot(A, xt)

#xk[1600 x 1]
# random = np.random.randint(-1, 1, size=(A.shape[1],))
# x = random.reshape(-1, 1)
x = np.zeros((A.shape[1], 1))

#rk[1600 x 1]
r = b - np.dot(A, x)

#ak
a = 0

#tolerance
E = 10E-10

#=========================================================================================================================================================================================

def get_next_a():
    return (np.dot(r.T, r) / np.dot(r.T, np.dot(A, r))).item()

def get_next_x():
    return x + r * a

def get_next_r():
    return r - np.dot(A, r) * a

def get_error():
    return np.linalg.norm(r) / np.linalg.norm(b)

#=========================================================================================================================================================================================

i = 0
x_axis = []
y_axis = []
x_axis.append(i)
y_axis.append(get_error())
while(get_error() > E):
    i += 1
    x_axis.append(i)

    a = get_next_a()
    x = get_next_x()
    r = get_next_r()

    y_axis.append(get_error())

#print k
print("Took", i, "iterations")

#write error into the txt file
with open('relative_residual.txt', 'w') as file:
    file.write(','.join(map(str, y_axis)))

plt.plot(x_axis, y_axis)
plt.yscale('log')
plt.show()

