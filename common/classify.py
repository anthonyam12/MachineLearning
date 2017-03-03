from math import *

'''
	Returns the Euclidian distance between two vectors

	x_1, x_2 - vectors (lists of values, must have same dimensions)

	NOTES: slightly test, should be correct
'''
def euclidianDistance(x_1, x_2):
	e = 0
	# start at 1 to skip the classification 
	for i in range(len(x_1)):
		e += (float(x_1[i]) - float(x_2[i]))**2
	return sqrt(e)

'''
	Takes a new vector of data and a training (test) set and returns a classification of the new data. 
	Compares the new vector with each test vector finding the k nearest neighbers, averages the classification, 
	of the nearest neighbors, and assigns that classification.

	number_classes is the total number of classifications that exist - 0-based

	!!!!!Index 0 of x_t and the training data must be the classification!!!!!
'''
def knn(x_t, training_data, number_classes, k=5):
	# large values for original neighbors [classification, distance]
	nn = []
	for i in range(k):
		nn.append([number_classes+1, 9999999])

	# find nearest neighbors
	for key in training_data:
		row = training_data[key]
		dist = euclidianDistance(row[1:], x_t)
		for d in nn:
			if dist < d[1]:
				d[1] = dist
				d[0] = row[0]
				break

	# average nearest neighbors classification
	avg_dist = 0
	min_dist = 9999999999
	votes = [0 for x in range(0, number_classes)]
	for neighbor in nn:
		votes[neighbor[0]] += 1
		if neighbor[1] < min_dist:
			min_dist = neighbor[1]
		avg_dist += neighbor[1]

	c = votes.index(max(votes))
	certainty = votes[c] / k

	return c, certainty, min_dist, avg_dist/k