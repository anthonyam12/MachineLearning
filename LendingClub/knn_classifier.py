import data_controller as dc
from math import *

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
		home_ownership, grade, emp_length, loan_status
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
		
		
		# loan_status
		ls = row[LOAN_STATUS]
		if ls == "Fully Paid" or ls == "Current":
			row[LOAN_STATUS] = FULLY_PAID
		else:
			row[LOAN_STATUS] = CHARGED_OFF
		
	return data_dict

'''
	Returns the Euclidian distance between two vectors

	x_1, x_2 - vectors (lists of values, must have same dimensions)

	x_1 will be the training vector so x_1[0] is the classification, therefore
		we need to offset x_1 by 1 place but start x_2 at 0
	NOTES: slightly test, should be correct
'''
def euclidianDistance(x_1, x_2):
	e = 0
	# start at 1 to skip the classification 
	for i in range(1, len(x_1)):
		e += (float(x_1[i]) - float(x_2[i-1]))**2
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
		dist = euclidianDistance(row, x_t)
		for d in nn:
			if dist < d[1]:
				d[1] = dist
				d[0] = row[LOAN_STATUS]
				break
	
	print(nn)
	# average nearest neighbors classification
	ones = 0
	zeros = 0
	for n in nn:
		if n[0] == 1:
			ones += 1
		else:
			zeros += 1

	if ones > zeros:
		return 1
	else:
		return 0


#######################################################################################################################


if __name__ == '__main__':
	cols = ['loan_status', 'loan_amnt', 'home_ownership', 'dti', 'int_rate', 'annual_inc', 'grade', 'emp_length', 'delinq_2yrs']

	print('Getting test data...')
	test_data = dc.get_data('LoanStats3a.csv', prune_cols=cols)
	test_data = intRateToNumeric(test_data)
	test_data = convertStringsToNumeric(test_data)
	test_data = removeBadRows(test_data)

	print('\nGetting train data...')
	train_data = dc.get_data('LoanStats_2016Q3.csv', prune_cols=cols)
	train_data = convertStringsToNumeric(train_data)
	train_data = intRateToNumeric(train_data)
	train_data = removeBadRows(train_data)

	# [:] makes a copy of a list by value
	right = 0
	wrong = 0
	total_cases = len(test_data)
	for key in test_data:
		row = test_data[key]
		actual = row[LOAN_STATUS]
		if knn(row[1:], train_data) == actual:
			right += 1
		else:
			wrong += 1

	print("Right: ", right/total_cases)
	print("Wrong: ", wrong/total_cases)
			
		


