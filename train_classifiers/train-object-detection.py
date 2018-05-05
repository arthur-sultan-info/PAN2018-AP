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
	labelToIntCount = 0

	for observation in X_train:
		for objectLabel in observation['labels']:
			if objectLabel not in convertLabelToInt:
				convertLabelToInt[objectLabel] = labelToIntCount
				labelToIntCount += 1
			columns.append(convertLabelToInt[objectLabel])
			rows.append(rowIndex)
			values.append(observation['labels'][objectLabel])
		
		rowIndex += 1 # next observation item

	row  = np.array(rows)
	col  = np.array(columns)
	data = np.array(values)
	numberOfRows = len(X_train)
	numberOfColumns = len(convertLabelToInt)
	resultSparseMatrix = csr_matrix((data, (row, col)), shape=(numberOfRows, numberOfColumns))

	return (resultSparseMatrix, convertLabelToInt)
	
	
def train(options):
	import pickle

	# Loading yolo labels
	author_images_yolo = dict()
	with open(options['features_path'] , "rb" ) as input_file:
		author_images_yolo = pickle.load(input_file)
	
	# Loading the truth file, for each language
	gender_dict = parse_gender_dict(options['dataset_path'] + '/ar/ar.txt')
	gender_dict.update(parse_gender_dict(options['dataset_path'] + '/en/en.txt'))
	gender_dict.update(parse_gender_dict(options['dataset_path'] + '/es/es.txt'))
	
	
	# Creating a dict with an incremental id as key and the array [features, author_gender] as value, for each image
	dataset = dict()
	i=0
	for author in author_images_yolo:
		author_id = author
		author_gender = gender_dict[author_id]
		for imageIndex in author_images_yolo[author_id]:
			# Dict of objects recognized by yolo
			features = author_images_yolo[author_id][imageIndex]
			
			dataset[i] = [features, author_gender]
			
			i += 1
	
	# Getting X and Y vectors from the dataset
	import pandas as pd
	
	dataset_array = np.asarray(list(dataset.values()))
	images_tweets = pd.DataFrame(dataset_array, columns=['features', 'label',])
	X = np.array(list(images_tweets.features))
	y = np.array(list(images_tweets.label), dtype=">i1")
	
	# Converting X to sparse matrix
	X, convertLabelToInt = X_ToSparseMatrix(X)
	
	# Selecting the best classifier from a 20-fold cross validation
	from sklearn.model_selection import cross_val_score
	from sklearn.naive_bayes import MultinomialNB
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.calibration import CalibratedClassifierCV
	from sklearn.svm import LinearSVC
	
	print('Training classifier for object detection -- This may take some time..')
	
	classifiers = [MultinomialNB(), RandomForestClassifier(random_state=0, n_estimators=10, n_jobs=-1), DecisionTreeClassifier(random_state=0), LinearSVC(random_state=0)]
	
	best_classifier = None
	best_accuracy = 0
	for clf in classifiers:
		current_scores = cross_val_score(clf, X, y, cv=20, scoring='accuracy', n_jobs=-1, verbose=1)
		if current_scores.mean() > best_accuracy:
			best_accuracy = current_scores.mean()
			best_classifier = clf
	
	
	# Training the classifier on the whole data
	if str(type(best_classifier)) == "<class 'sklearn.svm.classes.LinearSVC'>":
		best_classifier = CalibratedClassifierCV(best_classifier)
	best_classifier.fit(X, y)
	
	
	# Saving the classifier
	pickle.dump( best_classifier, open( "trained-classifiers/object-detection-classifier.p", "wb" ) )
	pickle.dump( convertLabelToInt, open( "./labels_object_detection/convertLabelToInt.p", "wb" ) )
	
	print('Best classifier saved for object detection:', best_classifier)
	print('Best accuracy:', best_accuracy)
	
	



if __name__ == "__main__":
	options = {
		'features_path': "../feature_extractors/extracted-features/author_images_yolo-low-train.p",
		'dataset_path': "../Min dataset"
	}
	
	train(options)