import csv


'''
	get_data(filename, prune=False)
	
	Returns the data in the csv file (filename) as s dictionary with the headers as the keys.

	filename - name of the csv file to read
	prune - determines if we should call the prune function to get the rows we want to analyze
'''
def get_data(filename, prune_cols=False):
	data_dict = dict()
	cols = []

	print("Opening file: ", filename)
	csvfile = open(filename, newline='', encoding="ISO-8859-1")
	reader = csv.DictReader(csvfile, )
	
	if prune_cols:
		cols = prune_cols
	else:
		cols = reader.fieldnames

	print("Retrieving columns: ", cols)	
	i = 0
	for row in reader:
		data_dict[i] = [row[col] for col in cols]
		i = i + 1

	return data_dict
