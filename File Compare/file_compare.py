import numpy as np




def read_file(filename):
	with open(filename,'r') as f_handle:
		file_data = f_handle.read().splitlines()
	return file_data
	
def append_pad(message):
	org_length = len(message)
	message = bytearray(message.encode('hex'))
	message.append(8)
	while len(message)%64 != 56:
		message.append(0)
	t = org_length & 8*bytearray(0)
	print(len(message))
	print(message)
	#~ bin_message += '1'

	
def main():
	x = read_file('in.txt')
	print(x)
	return 0
	
if __name__ == '__main__':
	#~ main()
	append_pad('zxv adfghsh  sgfj hbns')
	
	
