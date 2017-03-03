import os
import sys
import random

cwd = os.getcwd()
sys.path.append(cwd + '/../common/')
import data_controller as dc
import classify as clsfy


# CLASSES
IRIS_SETOSA = [0, 'Iris-setosa']
IRIS_VERSICOLOR = [1, 'Iris-versicolor']
IRIS_VIRGINICA = [2, 'Iris-virginica']

'''
	Prints user options.
'''
def print_options():
	print('\nSelect from the following options:')
	print('\t(N)ew measurements')
	print('\t(E)xit')

def get_iris_data():
	SepalLengthCm = float(input('Sepal Length (cm): '))
	SepalWidthCm = float(input('Sepal Width (cm): '))
	PetalLengthCm = float(input('Petal Length (cm): '))
	PetalWidthCm = float(input('Petal Width (cm): '))
	return [SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]


'''
	Allows users to input new measurements to classify the plants.
'''
def main_loop(neighbors, training_data):
	while True:
		print_options()
		choice = input('Select an option: ')
		if choice.lower() == 'e':
			break
		elif choice.lower() == 'n':
			x_t = get_iris_data()
			classification = clsfy.knn(x_t, training_data, 3, neighbors)[0]
			if classification == IRIS_SETOSA[0]:
				print('Classification: ', IRIS_SETOSA[1])
			elif classification == IRIS_VERSICOLOR[0]:
				print('Classification: ', IRIS_VERSICOLOR[1])
			elif classification == IRIS_VIRGINICA[0]:
				print('Classification: ', IRIS_VIRGINICA[1])
		else:
			print('Invalid option.')
	return


'''
	Converts the classification (a string) to a numeric value.
'''
def ClassToNumeric(data):
	for key in data:
		if data[key][0] == 'Iris-virginica':
			data[key][0] = IRIS_VIRGINICA[0]
		elif data[key][0] == 'Iris-versicolor':
			data[key][0] = IRIS_VERSICOLOR[0]
		elif data[key][0] == 'Iris-setosa':
			data[key][0] = IRIS_SETOSA[0]
			
	return data

if __name__ == "__main__":
	filename = "Iris.csv"
	data = dc.get_data(filename)

	neighbors = 3
	if len(sys.argv) > 1:
		neighbors = int(sys.argv[1])	

	# in this case, to shuffle the data we added a column in Excel that was just '=rand()'
	# dragging this column down prroduced a random number for each row, sorting on this
	# random column shuffled the data to be better split for test/train. Here we're removing
	# this dummy column from our model. Also modified CSV so classes are in column 0
	for key in data:
		row = data[key]
		data[key] = row[:-1]

	data = ClassToNumeric(data)

	train = dict()
	for k, v in list(data.items())[:-100]:
		train[k] = v

	test = dict()
	for k, v in list(data.items())[-100:]:
		test[k] = v
	
	right = 0
	wrong = 0
	correct_avg_dists = []
	correct_min_dists = []
	incorrect_min_dists = []
	incorrect_avg_dists = []
	for key in test:
		row = test[key]
		actual_class = row[0]
		class_data = clsfy.knn(row[1:], train, 3, neighbors)
		predicted_class = class_data[0]
		if actual_class != predicted_class:
			wrong += 1
			incorrect_min_dists.append(class_data[1])
			incorrect_avg_dists.append(class_data[2])
		else:
			right += 1
			correct_min_dists.append(class_data[1])
			correct_avg_dists.append(class_data[2])


	print('\n\nNeighbors: ', neighbors)

	print("Correct: ", right, '/', len(test), ' = ', right/len(test))
	print("Wrong: ", wrong, '/', len(test), ' = ', wrong/len(test))

	print('Incorrect Average Min Distance: ', sum(incorrect_min_dists)/len(incorrect_min_dists))
	print('Incorrect Average Average Distance: ', sum(incorrect_avg_dists)/len(incorrect_avg_dists))

	print('Correct Average Min Distance: ', sum(correct_min_dists)/len(correct_min_dists))
	print('Correct Average Min Distance: ', sum(correct_avg_dists)/len(correct_avg_dists))

	print('\n\nPredicted Iris\' with ', (right/len(test))*100, '% accuracy.')
	
	main_loop(neighbors, train)