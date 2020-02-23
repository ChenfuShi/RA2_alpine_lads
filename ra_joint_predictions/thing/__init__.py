import os

def print_train_dir_test():
	print('-----------------')
	print('Train Dir contents:')

	for file in os.listdir( '/train' ):
		print(file)

	print('-----------------')
