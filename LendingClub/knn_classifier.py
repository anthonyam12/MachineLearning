import data_controller as dc
from math import *
import threading
from queue import Queue
import random
import pylab as plt
import sys 

### GLOBAL STRING TO INT CONSTANTS
# NaN
NA = 99999999

# Classifications - Loan Status
CHARGED_OFF = 0
FULLY_PAID = 1

# Home Ownership
RENT = 2
MORTGAGE = 3
OWN = 4
OTHER = 1
NONE = 0

# Employment Length
TEN_PLUS = 10
NINE = 9
EIGHT = 8
SEVEN = 7 
SIX = 6
FIVE = 5
FOUR = 4
THREE = 3
TWO = 2
ONE = 1
LT_ONE = 0

# Grade
A = 0
B = 1
C = 2
D = 3
E = 4
F = 5
G = 6

# Term
MONTH_36 = 36
MONTH_60 = 60

# Column Numbers
LOAN_STATUS = 0
LOAN_AMNT = 1
HOME_OWNERSHIP = 2
DTI = 3
INT_RATE = 4
ANNUAL_INC = 5
GRADE = 6
EMP_LENGTH = 7
DELINQ_2YRS = 8
TERM = 9
INSTALLMENT = 10
INQ_LAST_6MTHS = 11
OPEN_ACC = 12
PUB_REC = 13
REVOL_BAL = 14
TOTAL_ACC = 15

#######################################################################################################################

# Functions
'''
	Converts the interest rate of the form 'xx%' into a numeric value between 0 and 100
'''
def intRateToNumeric(data_dict):
	for key in data_dict:
		str_value = str(data_dict[key][INT_RATE]) 
		data_dict[key][INT_RATE] = str_value.strip()[:-1]
	return data_dict

'''
	Removes rows in the data dictionary that may be shifted or otherwise bad.
'''
def removeBadRows(data_dict):
	bad_keys = []
	start_rows = len(data_dict)

	for key in data_dict:
		row = data_dict[key]
		if row[TOTAL_ACC] is None:
			bad_keys.append(key)
		try:
			float(row[LOAN_AMNT])
			float(row[DTI])
			float(row[INT_RATE])
			float(row[ANNUAL_INC])
			float(row[DELINQ_2YRS])
		except ValueError:
			bad_keys.append(key)

	for key in bad_keys:
		del(data_dict[key])
	print("Removed ", len(bad_keys), " out of ", start_rows, " rows.")

	return data_dict

'''
	Converts the previously converted string values back into int values (undo for 
		convertStringsToNumeric(...)) 
'''
def convertNumericToString(data_dict):
	
	return data_dict

'''
	Coverts the following quantitative (string) values to qualitative (integer) values:
		home_ownership, grade, emp_length, loan_status, term
'''
def convertStringsToNumeric(data_dict): 
	for key in data_dict:
		row = data_dict[key]

		# home ownership
		ho = row[HOME_OWNERSHIP]
		if ho == "RENT":
			row[HOME_OWNERSHIP] = RENT
		elif ho == "MORTGAGE":
			row[HOME_OWNERSHIP] = MORTGAGE
		elif ho == "OTHER":
			row[HOME_OWNERSHIP] = OTHER
		elif ho == "OWN":
			row[HOME_OWNERSHIP] = OWN
		elif ho == "NONE":
			row[HOME_OWNERSHIP] = NONE
		else:
			row[HOME_OWNERSHIP] = NA

		# grade
		grade = row[GRADE]
		if grade == "A":
			row[GRADE] = A
		elif grade == "B":
			row[GRADE] = B
		elif grade == "C":
			row[GRADE] = C
		elif grade == "D":
			row[GRADE] = D
		elif grade == "E":
			row[GRADE] = E
		elif grade == "F":
			row[GRADE] = F
		elif grade == "G":
			row[GRADE] = G
		else:
			row[GRADE] = NA

		# emp_length
		el = row[EMP_LENGTH]
		if el == "< 1 year":
			row[EMP_LENGTH] = LT_ONE
		elif el == "1 year":
			row[EMP_LENGTH] = ONE
		elif el == "2 years":
			row[EMP_LENGTH] = TWO
		elif el == "3 years":
			row[EMP_LENGTH] = THREE
		elif el == "4 years":
			row[EMP_LENGTH] = FOUR
		elif el == "5 years":
			row[EMP_LENGTH] = FIVE
		elif el == "6 years":
			row[EMP_LENGTH] = SIX
		elif el == "7 years":
			row[EMP_LENGTH] = SEVEN
		elif el == "8 years":
			row[EMP_LENGTH] = EIGHT
		elif el == "9 years":
			row[EMP_LENGTH] = NINE
		elif el == "10+ years":
			row[EMP_LENGTH] = TEN_PLUS
		else:
			row[EMP_LENGTH] = NA

		# term
		t = row[TERM]
		if t == "36 months":
			row[TERM] = MONTH_36
		elif t == "60 months":
			row[TERM] = MONTH_60
		else:
			row[TERM] = NA
		
		# loan_status
		ls = row[LOAN_STATUS]
		if ls == "Fully Paid" or ls == "Current":
			row[LOAN_STATUS] = FULLY_PAID
		else:
			row[LOAN_STATUS] = CHARGED_OFF
	
	return data_dict

'''
	Normalizes the values so that each feature has an equal vote on 
	'closeness'. Convert data to values between 0 and 1
'''
def normalize(data_dict):
	maxes = dict()
	
	# known/defined
	maxes[GRADE] = 6
	maxes[EMP_LENGTH] = 10
	maxes[TERM] = 60
	maxes[HOME_OWNERSHIP] = 4

	# unknown
	maxes[DTI] = 0
	maxes[INT_RATE] = 0
	maxes[ANNUAL_INC] = 0
	maxes[LOAN_AMNT] = 0
	maxes[DELINQ_2YRS] = 0
	maxes[INSTALLMENT] = 0
	maxes[INQ_LAST_6MTHS] = 0
	maxes[OPEN_ACC] = 0
	maxes[PUB_REC] = 0
	maxes[REVOL_BAL] = 0
	maxes[TOTAL_ACC] = 0

	for key in data_dict:
		row = data_dict[key]
		for i in range(1, 16):
			if float(row[i]) > maxes[i]:
				maxes[i] = float(row[i])

	for key in data_dict:
		row = data_dict[key]
		# normalize the values row[val] /= maxes[val]
		for i in range(1, 16):
			row[i] = float(row[i]) / maxes[i]

	return data_dict

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
'''
def knn(x_t, data_dict, k=5):
	# large values for original neighbors [classification, distance]
	nn = []
	for i in range(k):
		nn.append([100, 999999999])

	# find nearest neighbors	
	for key in data_dict:
		row = data_dict[key]
		dist = euclidianDistance(row[1:], x_t)
		for d in nn:
			if dist < d[1]:
				d[1] = dist
				d[0] = row[LOAN_STATUS]
				break
	
	#print(nn)
	# average nearest neighbors classification
	ones = 0
	zeros = 0
	min_dist = 999999999
	avg_dist = 0
	for n in nn:
		# weighted distance, the nearest neighbors with shortest distances count more
		# this weights the votes, not the features
		if n[0] == 1:
			ones += n[1]
		else:
			zeros += n[1]
		avg_dist += n[1]
		if n[1] < min_dist:
			min_dist = n[1]
			
	# want the ones/zeros with the smallest distance (these nodes were typically closer to x_t)
	if ones < zeros:
		certainty = ones/k
		return 1, certainty, min_dist, avg_dist
	else:
		certainty = zeros/k
		return 0, certainty, min_dist, avg_dist

'''
	Runs the program for a certain subset of the test_data.
	Mainly used for testing, when in production, very few vectors will be passed to the program.
'''
def worker(test_data, train_data, k=5):
	right = 0
	wrong = 0
	
	for key in test_data:
		row = test_data[key]
		actual = row[LOAN_STATUS]
		if knn(row[1:], train_data) == actual:
			right += 1
		else:
			wrong += 1
		print(key)
	q.put(right, wrong)

'''
	Tests the classifier using another data set.
'''
def test(train_data, test_data):
	total_cases = len(test_data)
	num_threads = 12
	dict_args = []
	rows_per_thread = int(len(test_data) / num_threads)
	for i in range(num_threads - 1):
		d = dict()
		start = i * rows_per_thread
		end = (start + rows_per_thread) - 1
		for k, v in list(test_data.items())[start:end]:
			d[k] = v
		dict_args.append(d)

	start = 7 * rows_per_thread
	d = dict()
	for k, v in list(test_data.items())[start:]:
		d[k] = v
	dict_args.append(d)
		

	# [:] makes a copy of a list by value
	right = 0
	wrong = 0

	q = Queue()
	threads = []
	for i in range(num_threads):
		t = threading.Thread(target=worker, args=(dict_args[i], train_data), name=str(i))
		t.daemon = True
		t.start()
		threads.append(t)	

	print(len(threads))

	for t in threads:
		t.join()

	print(q)

	for r in q:
		print(r)
		right += r[0]
		wrong += r[1]

	print("Right: ", right/total_cases)
	print("Wrong: ", wrong/total_cases)

'''
	Returns the training and test data.
'''
def getData(test_data='LoanStats3b.csv', train_data='train_data.csv'):
	cols = ['loan_status', 'loan_amnt', 'home_ownership', 'dti', 'int_rate', 'annual_inc', 'grade', 'emp_length', 'delinq_2yrs',
			'term', 'installment', 'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'total_acc']

	print('Getting test data...')
	test_data = dc.get_data(test_data, prune_cols=cols)
	test_data = intRateToNumeric(test_data)
	test_data = convertStringsToNumeric(test_data)
	test_data = removeBadRows(test_data)

	print('\nGetting train data...')
	train_data = dc.get_data(train_data, prune_cols=cols)
	train_data = convertStringsToNumeric(train_data)
	train_data = intRateToNumeric(train_data)
	train_data = removeBadRows(train_data)

	print("\nNormalizing data...")
	train_data = normalize(train_data)
	test_data = normalize(test_data)

	return test_data, train_data


'''
	Runs the classifier.
'''
def run(cases, neighbors, test_data, train_data):
	# get cases  
	start = random.randint(0, len(test_data) - (cases + 1))
	examples = dict()
	for k, v in list(test_data.items())[start:start+cases]:
		examples[k] = v

	right = 0
	wrong = 0
	i = 1
	correct_certain = []
	incorrect_certain = []
	positives = 0
	negatives = 0
	correct_avg_dist = []
	incorrect_avg_dist = []
	correct_min_dist = []
	incorrect_min_dist = []

	for key in examples:
		row = examples[key]
		actual = row[0]
		classification = knn(row[1:], train_data, neighbors)
		if classification[0] == 0:
			negatives += 1
		else:
			positives += 1
		if  classification[0] == actual:
			right += 1
			#print("Correct classification. Average Distance: ", classification[3])
			correct_certain.append(classification[1])
			correct_min_dist.append(classification[2])
			correct_avg_dist.append(classification[3])
		else:
			#print("Incorrect classification. Average Distance: ", classification[3], ' Actual: ', actual)
			incorrect_certain.append(classification[1])
			incorrect_min_dist.append(classification[2])
			incorrect_avg_dist.append(classification[3])
			wrong += 1
		i += 1
	
	print("\n\nRight: ", right, '/', cases, ' = ', right/cases)
	print("Wrong: ", wrong, '/', cases, ' = ', wrong/cases)

	print("\nCorrect Certainty: ", sum(correct_certain)/len(correct_certain))
	print("Incorrect Certainty: ", sum(incorrect_certain)/len(incorrect_certain))

	print("\nPositives: ", positives, '/', cases)
	print("Negatives: ", negatives, '/', cases)

	print("\nCorrect Min. Distance Average: ", sum(correct_min_dist)/len(correct_min_dist))
	print("Incorrect Min. Distance Average: ", sum(incorrect_min_dist)/len(incorrect_min_dist))

	print("\nCorrect Avg Distance Average: ", sum(correct_avg_dist)/len(correct_avg_dist))
	print("Incorrect Avg Distance Average: ", sum(incorrect_avg_dist)/len(incorrect_avg_dist))

	# c_run = [i for i in range(len(correct_avg_dist))]
	# i_run = [i for i in  range(len(incorrect_avg_dist))]
	
	# plt.plot(c_run, correct_avg_dist, 'g.')
	# plt.plot(i_run, incorrect_avg_dist, 'r.')
	# plt.show()
	return [right/cases, wrong/cases]

'''
	Entry of the program.

	Takes two parameters 'python3 knn_classifier.py <runs> <neighbors> <cases>' 
	OR 
	no paramters, defaults will be used:
		runs = 10
		neighbors = 5
		cases = 100
'''
if __name__ == '__main__':
	test_data, train_data = getData()

	if len(sys.argv) < 4:
		runs = 10
		neighbors = 5
		cases = 100
	else:
		runs = int(sys.argv[1])
		neighbors = int(sys.argv[2])
		cases = int(sys.argv[3])

	right = []
	wrong = []
	for i in range(1, runs + 1):
		print('\nProcessing run ' + str(i) + ' of ' + str(runs))
		r, w = run(cases, neighbors, test_data, train_data)
		right.append(r)
		wrong.append(w)

	print(right)
	print(wrong)
	