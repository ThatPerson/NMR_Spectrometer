import numpy as np

def ReducedWigner(j, k, l, beta):
	if (j == 0):
		d = 1
	elif (j == 1):
		if (k == 0 and l == 0):
			d = np.cos(beta);
		elif ((k == 0 and l == 1) or (k == -1 and l == 0)):
			d = np.sin(beta)/np.sqrt(2);
		elif ((k == 0 and l == -1) or (k == 1 and l == 0)):
			d = -np.sin(beta)/np.sqrt(2);
		elif ((k == 1 and l == -1) or (k == -1 and l == 1)):
			d = (1 - np.cos(beta))/2;
		elif ((k == 1 and l == 1) or (k == -1 and l == -1)):
			d = (1 + np.cos(beta))/2;
	elif (j == 2):
		if (k == 0 and l == 0):
			d = (3 * np.power(np.cos(beta), 2) - 1)/2;
		elif ((k == 1 and l == 0) or (k == 0 and l == -1)):
			d = -np.sqrt(3/2) * np.sin(beta)*np.cos(beta);
		elif ((k == 0 and l == 1) or (k == -1 and l == 0)):
			d = np.sqrt(3/2) * np.sin(beta)*np.cos(beta);
		elif ((k == 1 and l == -1) or (k == -1 and l == 1)):
			d = (2 * np.cos(beta) + 1) * (1 - np.cos(beta))/2;
		elif ((k == 1 and l == 1) or (k == -1 and l == -1)):
			d = (2 * np.cos(beta) - 1) * (1 + np.cos(beta))/2;
		elif ((k == 2 and l == 0) or (k == 0 and l == 2) or (k == -2 and l == 0) or (k == 0 and l == -2)):
			d = np.sqrt(3/8) * np.power(np.sin(beta), 2);
		elif ((k == 2 and l == 1)):
			d = -np.sin(beta)*(np.cos(beta)+1)/2;
		elif ((k == 1 and l == 2) or (k == -2 and l == -1) or (k == -1 and l == -2)):
			d = np.sin(beta)*(np.cos(beta)+1)/2;
		elif ((k == 2 and l == -1) or (k == 1 and l == -2)):
			d = np.sin(beta)*(np.cos(beta) - 1)/2;
		elif ((k == -2 and l == 1) or (k == -1 and l == 2)):
			d = -np.sin(beta)*(np.cos(beta)-1)/2;
		elif ((k == 2 and l == 2) or (k == -2 and l == -2)):
			d = np.power(np.cos(beta/2), 4);
		elif ((k == 2 and l == -2) or (k == -2 and l == 2)):
			d = np.power(np.sin(beta/2), 4);
	return d

def WignerD(ji, k, l, c):
	alpha = c[0]
	beta = c[1]
	gamma = c[2]
	d = np.exp(-1j * k * alpha) * ReducedWigner(ji, k, l, beta) * np.exp(-1j * l * gamma);
	return d
