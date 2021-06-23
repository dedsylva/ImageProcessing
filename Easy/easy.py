import sys
import os

def main(argv):
	data = [a.split('=') for a in argv]

	for d in data:
		if d[0] == 'model':
			if d[1] == 'MNIST':
				os.system('python MNIST/main.py')
			elif d[1] == 'FMNIST':
				os.system('python FMNIST/main.py')
			elif d[1] == 'animals':
				print('Not done Yet :(')
				continue
			else:
				raise Exception ('The Model you entered are not available')

if __name__ == '__main__':
	main(sys.argv[1:])