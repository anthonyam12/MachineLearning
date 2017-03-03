import random
import os
import sys

cwd = os.getcwd()
sys.path.append(cwd + '/../common/')
import data_controller as dc
import classify as clsfy


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


#######################################################################################################################


if __name__ == '__main__':
	cols = ['loan_status', 'loan_amnt', 'home_ownership', 'dti', 'int_rate', 'annual_inc', 'grade', 'emp_length', 'delinq_2yrs',
			'term', 'installment', 'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'total_acc']

	print('Getting test data...')
	test_data = dc.get_data('LoanStats3b.csv', prune_cols=cols)
	test_data = intRateToNumeric(test_data)
	test_data = convertStringsToNumeric(test_data)
	test_data = removeBadRows(test_data)

	print('\nGetting train data...')
	train_data = dc.get_data('train_data.csv', prune_cols=cols)
	train_data = convertStringsToNumeric(train_data)
	train_data = intRateToNumeric(train_data)
	train_data = removeBadRows(train_data)

	# get 100 rows 
	cases = 300
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
		classification = clsfy.knn(row[1:], train_data, 2, 3)
		if classification[0] == 0:
			negatives += 1
		else:
			positives += 1
		if  classification[0] == actual:
			right += 1
			print("Correct classification. Certainty: ", classification[1])
			correct_certain.append(classification[1])
			correct_min_dist.append(classification[2])
			correct_avg_dist.append(classification[3])
		else:
			print("Incorrect classification. Certainty: ", classification[1], ' Actual: ', actual)
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
