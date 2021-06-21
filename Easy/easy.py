import sys

def main(argv):
	data = [a.split('=') for a in argv]

	for d in data:
		if d[0] == 'model':
			if d[1] == 'MNIST':
				continue
			elif d[1] == 'FMNIST':
				continue
			elif d[1] == 'animals':
				continue
			else:
				raise Exception ('The Model you entered are not available')

if __name__ == '__main__':
	main(sys.argv[1:])