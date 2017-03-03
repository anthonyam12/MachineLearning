import os
import sys
import random

cwd = os.getcwd()
sys.path.append(cwd + '/../common/')
import data_controller as dc
import classify as clsfy


# CLASSES
IRIS_SETOSA = 0
IRIS_VERSICOLOR = 1
IRIS_VIRGINICA = 2


'''
	Converts the classification (a string) to a numeric value.
'''
def ClassToNumeric(data):
	for key in data:
		if data[key][0] == 'Iris-virginica':
			data[key][0] = IRIS_VIRGINICA
		elif data[key][0] == 'Iris-versicolor':
			data[key][0] = IRIS_VERSICOLOR
		elif data[key][0] == 'Iris-setosa':
			data[key][0] = IRIS_SETOSA
			
	return data

if __name__ == "__main__":
	filename = "Iris.csv"
	data = dc.get_data(filename)
	neighbors = 5

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

	print("\nCorrect: ", right, '/', len(test), ' = ', right/len(test))
	print("Wrong: ", wrong, '/', len(test), ' = ', wrong/len(test))

	print('\nIncorrect Average Min Distance: ', sum(incorrect_min_dists)/len(incorrect_min_dists))
	print('Incorrect Average Average Distance: ', sum(incorrect_avg_dists)/len(incorrect_avg_dists))

	print('\nCorrect Average Min Distance: ', sum(correct_min_dists)/len(correct_min_dists))
	print('Correct Average Min Distance: ', sum(correct_avg_dists)/len(correct_avg_dists))
