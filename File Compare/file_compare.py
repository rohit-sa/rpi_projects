import numpy as np
import math



def read_file(filename):
	with open(filename,'r') as f_handle:
		file_data = f_handle.read().splitlines()
	return file_data
	
def append_pad(message):
	length = len(message)
	message = bytearray(message)
	message.append(0x80)
	while len(message)%64 != 56:
		message.append(0x00)
	len_pad = '%0.16x' % length
	for i in range(0,len(len_pad),2):
		message.append(int(len_pad[i:i+1], 16))
	print(len(message))
	return message

def F(x, y, z):
	return (x & y) | (~x & z)

def G(x, y, z):
	return (x & z) | (y & ~z)
	
def H(x, y, z):
	return (x ^ y ^ z)
	
def I(x, y, z):
	return (y ^ (x | ~z))

def k_table():
	K = np.zeros((64,1))
	for i in range(63):
		K[i] = math.floor(4294967296 * abs(math.sin(i+1)))
	return K

def md5(M):
	A = 0x67452301
	B = 0xefcdab89
	C = 0x98badcfe
	D = 0x10325476
	K = k_table()
	AA, BB, CC, DD = A, B, C, D
	X = []
	for i in range(0,64,4):
		X.append( M[i:i+4])
	#Round 1
	#
	print(X[0])
	
def main():
	x = read_file('in.txt')
	M = append_pad('adfgds')
	md5(M)
	return 0
	
if __name__ == '__main__':
	main()
