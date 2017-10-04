import numpy as np
import math
import struct
import hashlib

def read_file(filename):
	with open(filename,'r') as f_handle:
		file_data = f_handle.read().splitlines()
	return file_data

class MD5:
	
	def __init__(self):
		self.t_table()
		self.s = [7,12,17,22,5,9,14,20,4,11,16,23,6,19,15,21]
		self.X = []
	
	def append_pad(self,message):
		length = len(message)
		message = bytearray(message)
		message.append(0x80)
		
		while len(message)%64 != 56:
			message.append(0x00)
		len_pad = '%0.16x' % 16
		for i in range(0,len(len_pad),2):
			print(len_pad[i:i+2])
			message.append(int(len_pad[i:i+2], 16))
		print(len(message))
		return message

	def F(self, x, y, z):
		return (x & y) | (~x & z)
		#~ return np.bitwise_and(np.uint32(x), np.uint32(y)) | np.bitwise_and(np.uint32(~x), np.uint32(z))

	def G(self, x, y, z):
		return (x & z) | (~z & y)
		#~ return np.bitwise_and(np.uint64(x), np.uint64(z)) | np.bitwise_and(np.uint64(~z), np.uint64(y))
		
	def H(self, x, y, z):
		return x ^ y ^ z
		#~ return np.bitwise_xor(np.bitwise_xor(np.uint64(x),np.uint64(y)),np.uint64(z))
		
	def I(self, x, y, z):
		return y ^ ( x | ~z)
		#~ return np.bitwise_xor(np.uint64(y),np.bitwise_or(np.uint64(x), np.uint64(~z)))

	def t_table(self):
		self.T = np.zeros((64,1))
		for i in range(63):
			self.T[i] = math.floor(4294967296 * abs(math.sin(i+1)))

	def rol(self, x, n):
		x = np.uint32(x)
		#~ x = np.bitwise_and(np.uint32(0xFFFFFFFF), np.uint32(x))
		return np.uint32((x >> (32-n)) | (x << n)) & 0xFFFFFFFF

	def shift(self, lst, n):
		rot_lst = list(lst)
		if (not len(lst)):
			return
		n = n % len(lst)
		return rot_lst[len(lst)-n:] + rot_lst[:len(lst)-n]
	
	def round1(self,lst,k,s,i):
		a, b, c, d = lst
		a = b + self.rol(a + self.F(b,c,d) + self.X[k] + self.T[i], s)
		return a
	
	def round2(self,lst,k,s,i):
		a, b, c, d = lst
		a = b + self.rol(a + self.G(b,c,d) + self.X[k] + self.T[i], s)
		return a
		
	def round3(self,lst,k,s,i):
		a, b, c, d = lst
		a = b + self.rol(a + self.H(b,c,d) + self.X[k] + self.T[i], s)
		return a
		
	def round4(self,lst,k,s,i):
		a, b, c, d = lst
		a = b + self.rol(a + self.I(b,c,d) + self.X[k] + self.T[i], s)
		return a
		
	def md5_hash(self, M):
		self.T = np.asarray(self.T)
		M = self.append_pad(M)
		A = np.uint32(0x67452301)
		B = np.uint32(0xefcdab89)
		C = np.uint32(0x98badcfe)
		D = np.uint32(0x10325476)
		hash_list = np.asarray([A, B, C, D])
		AA, BB, CC, DD = A, B, C, D
		
		for i in range(0,64,4):
			self.X.append(struct.unpack('<L',M[i:i+4])[0])
			#~ print(struct.unpack('<L',M[i:i+4]))
		
		self.X = np.asarray(self.X)
		for i in range(0,64):
			#~ print(i)
			s_index = ((i//16)*4) +(i%4)
			rot_index = i%4
			hash_list = self.shift(hash_list,i)
			if(i >= 0 and i < 16):
				k = i
				hash_list[0] = self.round1(hash_list,k,self.s[s_index],i)
			elif(i >= 16 and i < 31):
				k = (5*i+1)%16
				hash_list[0] = self.round2(hash_list,k,self.s[s_index],i)
			elif(i >= 32 and i < 48):
				k = (3*i+5)%16
				hash_list[0] = self.round3(hash_list,k,self.s[s_index],i)
			else:
				k = (7*i+1)%16
				hash_list[0] = self.round4(hash_list,k,self.s[s_index],i)
		
		hash_list[0] += AA
		hash_list[1] += BB
		hash_list[2] += CC
		hash_list[3] += DD
		hash_value = ''
		for i in range(4):
			hash_value += ('%08X' % hash_list[i])
			#hash_value += str(hash_list[i])
		print(hash_value)	
		
def main():
	x = read_file('in.txt')
	m = MD5()
	m.md5_hash('ab')
	m_org = hashlib.md5()
	m_org.update('ab')
	print(m_org.hexdigest())
	return 0
	
if __name__ == '__main__':
	main()
	
