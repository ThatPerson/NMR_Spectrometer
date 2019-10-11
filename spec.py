import scipy as sp
import numpy as np
from scipy.linalg import expm 
from numpy.fft import fft
import sys
# define standard operators

if (len(sys.argv) <= 2):
    print("Please enter output filename")
    exit()

Ix = np.array([[0, 0.5], [0.5, 0]])
Iy = np.array([[0, -0.5j], [0.5j, 0]])
Iz = np.array([[0.5, 0], [0, -0.5]])
E = np.eye(2)

spins = []
num_spins = 0
J = []

B0 = 3

def m_kron(A):
    # A is a list of components in order of the spins. eg for spins A, B, C..., [E, E, Ix, E,..] would represent Cx.
    # Let N be the Kronecker product of A[0] and A[1]. Then for the range 2 to len(A), take the Kronecker product of this.
    if (len(A) <= 1):
        return A[0]
    N = np.kron(A[0], A[1])
    for i in range(2, len(A)):
        N = np.kron(N, A[i])
    return N



def gen_expm(H, t):
    # t can be either time (for Hamiltonian evolution) or pulse angle (eg for x pulse)
    factor = 1j * t * H
    return [expm(-factor), expm(factor)]

class NMR_Experiment:
    spins = []
    num_spins = 0
    J = 0
    P = 0
    def __init__(self):
        # eventually make this read in a file of some kind.
        self.spins.append(60)
        self.spins.append(120)
        self.spins.append(180)
        self.num_spins = len(self.spins)
        self.J = np.zeros((len(self.spins), len(self.spins)))
        self.J[0, 1] = 3 # only do one (so no 1,0)
        self.J[0, 2] = 0
        self.J[1, 2] = 1

    def operator(self, A):
        # A is a list of lists, each submember containing [spin, operator]. 
        # For example, Iz on spin 1 would be [[1, Iz]]. 
        L = []
        for i in self.spins:
            L.append(E)
        for i in A:
            L[i[0]] = i[1]
        return m_kron(L)

    def gen_hamiltonian(self):
        ham = 0
        for i in range(0, len(self.spins)):
            ham = ham + self.spins[i] * self.operator([[i, Iz]])
        for (x,y), v in np.ndenumerate(self.J):
            ham = ham + 2 * np.pi * v * (self.operator([[x, Ix], [y, Ix]]) + self.operator([[x, Iy], [y, Iy]]) + self.operator([[x, Iz], [y, Iz]]))
            print("Added %d-%d %d" % (x, y, v))
        return ham

    def apply_expm(self, ex):
        self.P = np.matmul(np.matmul(ex[0], self.P), ex[1])
        return self.P

    def apply_operator(self, Op, t):
        l = gen_expm(Op, t)
        return self.apply_expm(l)

Xmag = np.array(())
Ymag = np.array(())
time = np.array(())

penguin = NMR_Experiment()

for i in range(0, len(penguin.spins)):
    penguin.P = penguin.P + penguin.operator([[i, Iz]]) # initially have both spins along z
for i in range(0, len(penguin.spins)):
    penguin.apply_operator(penguin.operator([[i, Ix]]), np.pi/2)
effective_hamiltonian = penguin.gen_hamiltonian()
print(effective_hamiltonian)
dt = 1./100

eff_hamiltonian = gen_expm(effective_hamiltonian, dt)


for t in range(0, 10000):
    penguin.apply_expm(eff_hamiltonian)
    x = 0
    y = 0
    for i in range(0, len(penguin.spins)):
        x = x + np.trace(np.matmul(penguin.P, penguin.operator([[i, Ix]])))
        y = y + np.trace(np.matmul(penguin.P, penguin.operator([[i, Iy]])))
        
    Xmag = np.append(Xmag, x)
    Ymag = np.append(Ymag, y)
    time = np.append(time, t * dt)


with open(sys.argv[1], "w") as f:
    for i in range(0, len(time)):
        f.write("%f, %f, %f, %f, %f\n" % (time[i], np.real(Xmag[i]), np.imag(Xmag[i]), np.real(Ymag[i]), np.imag(Ymag[i])))
        

FK = np.abs(fft(Xmag + 1j * Ymag))
freq = np.arange(0, len(Xmag)) * (1. / (dt * len(Xmag))) * (50/8.)
with open(sys.argv[2], "w") as f:
    for i in range(0, len(FK)):
        f.write("%f, %f\n" % (freq[i], (FK[i])))
