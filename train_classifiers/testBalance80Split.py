import numpy as np

def parse_gender_dict(truthFilePath):
	with open(truthFilePath) as f:
		content = f.readlines()
		content = [x.strip() for x in content]
		
	genders = dict()
	# Female label is 0 ; Male label is 1
	for author_info in content:
		infos = author_info.split(':::')
		current_author_gender = None
		if(infos[1] == 'female'):
			current_author_gender = 0
		else:
			current_author_gender = 1
		genders[infos[0]] = current_author_gender
	
	return genders

from scipy.sparse import csr_matrix

def X_ToSparseMatrix(X_train):	
    columns = []
    rows = []
    values = []
    rowIndex = 0

    convertLabelToInt = dict()
    color_histogram_flattened_length = len(X_train[0]['local_binary_patterns'])

    for observation in X_train:
        observation_color_histogram = observation['local_binary_patterns']
        color_histogram_index = 0
        while (color_histogram_index < len(observation_color_histogram)):
            columns.append(color_histogram_index)
            rows.append(rowIndex)
            values.append(observation_color_histogram[color_histogram_index])
            color_histogram_index += 1
        
        rowIndex += 1 # next observation item
    
    row  = np.array(rows)
    col  = np.array(columns)
    data = np.array(values)
    numberOfRows = len(X_train)
    numberOfColumns = color_histogram_flattened_length + len(convertLabelToInt)
    resultSparseMatrix = csr_matrix((data, (row, col)), shape=(numberOfRows, numberOfColumns))
    
    return (resultSparseMatrix, convertLabelToInt)
	
	
def train(options):
	import pickle

	# Loading color histograms
	splitAuthor = dict()
	with open('../output/splitting/es.pkl' , "rb" ) as input_file:
		splitAuthor = pickle.load(input_file)
	
	# Loading the truth file, for each language
	gender_dict = parse_gender_dict(options['dataset_path'] + '/ar/ar.txt')
	gender_dict.update(parse_gender_dict(options['dataset_path'] + '/en/en.txt'))
	gender_dict.update(parse_gender_dict(options['dataset_path'] + '/es/es.txt'))
	
	
	# Creating a dict with an incremental id as key and the array [features, author_gender] as value, for each image
	nbMaleSplitA = 0
	nbFemaleSplitA = 0
	nbMaleSplitB = 0
	nbFemaleSplitB = 0
	for author in splitAuthor:
		if(splitAuthor[author] == 0):
			if(gender_dict[author] == 0):
				nbFemaleSplitA += 1
			else:
				nbMaleSplitA += 1
		else:
			if(gender_dict[author] == 0):
				nbFemaleSplitB += 1
			else:
				nbMaleSplitB += 1
	
	print('nbMaleSplitA',nbMaleSplitA)
	print('nbFemaleSplitA',nbFemaleSplitA)
	print('nbMaleSplitB',nbMaleSplitB)
	print('nbFemaleSplitB',nbFemaleSplitB)
	
	



if __name__ == "__main__":
	options = {
		'features_path': "../feature_extractors/extracted-features/author_images_global_features-low-train.p",
		'dataset_path': "../Min dataset"
	}
	
	train(options)