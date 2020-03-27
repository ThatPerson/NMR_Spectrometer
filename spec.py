import scipy as sp
import numpy as np
from scipy.linalg import expm
from numpy.fft import fft, fft2
import sys
import random
import wigner
# define standard operators

if (len(sys.argv) <= 4):
	print("Please enter commands: SIMFILE 1d/2d FIDFILE SPECFILE")
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

def isint(x):
	try:
		int(x)
		return 1
	except:
		return 0


class NMR_Experiment:
	spins = []
	num_spins = 0
	J = 0
	P = 0
	pulse_sequence = []
	acquire_on = []
	xmag = []
	ymag = []
	time = []
	crystallites = []
	spin_rate = 0
	solid = 0
	two_d = -1
	def __init__(self, filename):
		### files are of the format
		#SPINS
		#30
		#70
		#JCOUPLINGS
		#0, 1, 7
		#SEQUENCE
		#1.57[0x]
		#1.57[1x]
		#ACQ[0,1]
		self.two_d = -1
		self.load_file(filename)

		self.num_spins = len(self.spins)

#self.gen_crystallites(int(k[0]))
#					else:
#						self.load_crystallites(k[0])
						
	def gen_crystallites(self, x):
		for i in range(0, x):
			alph = np.pi * random.random()
			bet = np.arcsin((random.random() * 2) - 1) + np.pi/2
			gamm = np.pi * random.random()
			self.crystallites.append([alph, bet, gamm])
	
	def load_crystallites(self, f):
		with open(f, "r") as f:
			for l in f:
				try:
					k = [float(p) for p in l.split(",")]
				except:
					print("Error reading")
					continue
				if (len(k) != 3):
					continue
				self.crystallites.append(k)

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
		return ham

	def apply_expm(self, ex):
		self.P = np.matmul(np.matmul(ex[0], self.P), ex[1])
		return self.P

	def apply_operator(self, Op, t):
		l = gen_expm(Op, t)
		return self.apply_expm(l)

	def load_file(self, filename):
		mode = 0
		with open(filename, 'r') as fi:
			for i in fi:
				line = i.strip().replace(' ', '')
				if (line == "SPINS"):
					mode = 0
				elif (line == "JCOUPLINGS"):
					mode = 1
					self.num_spins = len(self.spins)
					self.J = np.zeros((self.num_spins, self.num_spins))
					for n in range(0, len(self.spins)):
						self.P = self.P + self.operator([[n, Iz]])
				elif (line == "DIPOLAR"):
					mode = 3
					self.dipolar = np.zeros((self.num_spins, self.num_spins))
				elif (line[:len("SETTINGS")] == "SETTINGS"):
					k = line[len("SETTINGS"):]
					k = k.split("/")
					if (len(k) != 2):
						print("Ignoring settings...")
						continue
					if (isint(k[0]) == 1):
						self.gen_crystallites(int(k[0]))
					else:
						self.load_crystallites(k[0])
					self.spin_rate = int(k[1])
					print("%s/%d" % (k[0], int(k[1])))
					self.solid = 1
				elif(line == "SEQUENCE"):
					mode = 2
				else:
					if (mode == 0):
						self.spins.append(float(line))
						print("Spin %d: %f" % (len(self.spins) - 1, float(line)))
					elif (mode == 1):
						p = line.split(",")
						if (len(p) < 3):
							print("Insufficient arguments for J coupling")
						else:
							self.J[int(p[0]), int(p[1])] = float(p[2])
							print("J Coupling from %d-%d=%fHz" % (int(p[0]), int(p[1]), float(p[2])))
					elif (mode == 3):
						p = line.split(",")
						if (len(p) < 3): # nuc1, nuc2, coupling constant
							print("Insufficient arguments for dipolar coupling")
						else:
							self.dipolar[int(p[0]), int(p[1])] = float(p[2])
							print("Dipolar Coupling from %d-%d=%fHz" % 
								(int(p[0]), int(p[1]), float(p[2])))
					elif (mode == 2):
						# of the form T[np, mq]
						if (line[:len("ACQ")] == "ACQ"):
							k = line[len("ACQ"):].replace("[", "").replace("]", "").split(",")
							for l in k:
								self.acquire_on.append(int(l))
						elif (line[:len("MIX")] == "MIX"):
							self.two_d = len(self.pulse_sequence)
						elif (line[:len("DELAY")] == "DELAY"):
							if (self.solid == 1):
								print("Error: DELAY not allowed for solid state.")
								exit()
							# add a time delay
							t_del = float(line[len("DELAY"):])
							e_ham = self.gen_hamiltonian()
							eff_h = gen_expm(e_ham, t_del)
							self.pulse_sequence.append(eff_h)
						else:
							k = line.split("[")
							if (len(k) < 2):
								print("Incorrectly formatted pulse")
							else:
								t = float(k[0])
								pl = k[1].replace("]", "")
								lp = pl.split(",")
								A = [] # becomes [[1, Iz], [2, Iz]] etc
								n = []
								for kx in lp:
									n = []
									n.append(int(kx[:-1]))
									if (kx[-1] == "x"):
										n.append(Ix)
									elif (kx[-1] == "y"):
										n.append(Iy)
									elif (kx[-1] == "z"):
										n.append(Iz)
									else:
										print("Incorrect pulse")
										continue
									A.append(n)

								self.pulse_sequence.append(gen_expm(self.operator(A), t))



	def run(self, ps = -1):
		# run the pulse sequence
		if (ps == -1):
			ps = self.pulse_sequence
		for i in ps:
			self.apply_expm(i)


	def acquire(self, steps, dt, retain = 1):
		e_hamiltonian = self.gen_hamiltonian()
		eff_ham = gen_expm(e_hamiltonian, dt)
		xmag = np.array(())
		ymag = np.array(())
		time = np.array(())
		prior_p = self.P
		for t in range(0, steps):
			self.apply_expm(eff_ham)
			x = 0
			y = 0
			for i in self.acquire_on:

				x = x + np.trace(np.matmul(self.P, self.operator([[i, Ix]])))
				y = y + np.trace(np.matmul(self.P, self.operator([[i, Iy]])))
			#x = x + random.randint(-10, 10)
			#y = y + random.randint(-10, 10)
			xmag = np.append(xmag, x)
			ymag = np.append(ymag ,y)
			time = np.append(time, t * dt)
		self.P = prior_p
		if (retain == 1):
			self.xmag = xmag
			self.ymag = ymag
			self.time = time
		return xmag, ymag, time

	def run_2d(self, steps_1, dt_1, steps_2, dt_2, retain = 1):
		self.run(self.pulse_sequence[:self.two_d])
		e_hamiltonian = self.gen_hamiltonian()
		eff_ham_2 = gen_expm(e_hamiltonian, dt_2)

		xmag_l = []
		ymag_l = []
		time_l = []
		prior_p = self.P
		for t1 in range(0, steps_1):
			self.P = prior_p
			eff_ham_1 = gen_expm(e_hamiltonian, dt_1 * t1)
			self.apply_expm(eff_ham_1)
			self.run(self.pulse_sequence[self.two_d:])
			xm, ym, ti = self.acquire(steps_2, dt_2, 0)
			xmag_l.append(xm)
			ymag_l.append(ym)
			time_l.append(ti)

		xmag = np.asarray(xmag_l)
		ymag = np.asarray(ymag_l)
		time = np.asarray(time_l)
		if (retain == 1):
			print("Retaining...")
			self.xmag = xmag
			self.ymag = ymag
			self.time = time
		return xmag, ymag, time

	def output_fid(self, fn):
		if (self.two_d == -1):
			with open(fn, "w") as f:
				for i in range(0, len(self.time)):
					f.write("%f, %f, %f, %f, %f\n" % (self.time[i],
						np.real(self.xmag[i]), np.imag(self.xmag[i]),
						np.real(self.ymag[i]), np.imag(self.ymag[i])))
		else:
			with open(fn, "w") as f:
				print(self.xmag.shape)
				for (x,y), v in np.ndenumerate(self.xmag):
					f.write("%f, %f, %f, %f, %f, %f\n" % (x, y, np.real(self.xmag[x, y]),
							np.imag(self.xmag[x, y]), np.real(self.ymag[x, y]),
							np.imag(self.ymag[x, y])))


	def output_spectrum(self, fn, dt):
		if (self.two_d == -1):
			FK = np.abs(fft(self.xmag + 1j * self.ymag))
			freq = np.arange(0, len(self.xmag)) * (1. / (dt * len(self.xmag))) * (50/8.)
			with open(fn, "w") as f:
				for i in range(0, len(FK)):
					f.write("%f, %f\n" % (freq[i], (FK[i])))
		else:
			FK = np.abs(fft2(self.xmag + 1j * self.ymag))
			with open(fn, "w") as f:
				for (x, y), v in np.ndenumerate(FK):
					f.write("%f, %f, %f\n" % (x, y, v))

filename = sys.argv[1]

if (sys.argv[2] == "1d"):
	penguin = NMR_Experiment(filename)
	penguin.run()
	penguin.acquire(10000, 1./100)
	penguin.output_fid(sys.argv[3])
	penguin.output_spectrum(sys.argv[4], 1./100)
else:
	penguin = NMR_Experiment(filename)
	penguin.run_2d(1000, 1./100, 1000, 1./100)
	#penguin.acquire(10000, 1./100)
	penguin.output_fid(sys.argv[3])
	penguin.output_spectrum(sys.argv[4], 1./50)
