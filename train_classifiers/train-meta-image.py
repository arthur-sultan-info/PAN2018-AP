import numpy as np
import pandas as pd

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

def X_ToSparseMatrix_objectDetection(X_train, convertLabelToInt):
    columns = []
    rows = []
    values = []
    rowIndex = 0

    labelToIntCount = 0
    
    for observation in X_train:
        for objectLabel in observation['labels']:
            if objectLabel in convertLabelToInt:
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
    
    return resultSparseMatrix, convertLabelToInt

def X_ToSparseMatrix_colorHistogram(X_train):	
    columns = []
    rows = []
    values = []
    rowIndex = 0

    convertLabelToInt = dict()
    color_histogram_flattened_length = len(X_train[0]['color_histogram'])

    for observation in X_train:
        observation_color_histogram = observation['color_histogram']
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

def X_ToSparseMatrix_lbp(X_train):	
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
	
def X_ToSparseMatrix_faceRecognition(X_train):	
	columns = []
	rows = []
	values = []
	rowIndex = 0

	convertLabelToInt = dict()
	labelToIntCount = 0

	for observation in X_train:
		if observation['Female'] > 0:
			columns.append(0)
			rows.append(rowIndex)
			values.append(observation['Female'])
		if observation['Male'] > 0:
			columns.append(1)
			rows.append(rowIndex)
			values.append(observation['Male'])
		
		rowIndex += 1 # next observation item

	row  = np.array(rows)
	col  = np.array(columns)
	data = np.array(values)
	numberOfRows = len(X_train)
	numberOfColumns = 2
	resultSparseMatrix = csr_matrix((data, (row, col)), shape=(numberOfRows, numberOfColumns))

	return (resultSparseMatrix, convertLabelToInt)
	
def createDataset(author_images, gender_dict):
	dataset = dict()
	for author in author_images:
		author_id = author
		author_gender = gender_dict[author_id]
		for imageIndex in author_images[author_id]:
			features = author_images[author_id][imageIndex]
			
			dataset[str(author_id + '.' + str(imageIndex))] = [features, author_gender]
	
	return dataset

def getXandYandIds(dataset):
	X = []
	y = []
	ids = []
	for author in dataset:
		X.append(dataset[author][0])
		y.append(dataset[author][1])
		ids.append(author)
	return X, y, ids
	
def train(options):
	import pickle

	# Loading object detection labels
	author_images_object_detection = dict()
	with open(options['features_path_yolo'] , "rb" ) as input_file:
		author_images_object_detection = pickle.load(input_file)
	
	# Loading face recognition labels
	author_images_face_recognition = dict()
	with open(options['features_path_face_recognition'] , "rb" ) as input_file:
		author_images_face_recognition = pickle.load(input_file)
	
	# Loading global features
	author_images_global_features = dict()
	with open(options['features_path_global_features'] , "rb" ) as input_file:
		author_images_global_features = pickle.load(input_file)
	
	
	# Concatening all features in a big author_images
	for author in author_images_object_detection:
		for image_index in author_images_object_detection[author]:
			author_images_object_detection[author][image_index].update(author_images_global_features[author][image_index])
			author_images_object_detection[author][image_index].update(author_images_face_recognition[author][image_index])
	author_images = author_images_object_detection
	
	# Loading the truth file, for each language
	gender_dict = parse_gender_dict(options['dataset_path'] + '/ar/ar.txt')
	gender_dict.update(parse_gender_dict(options['dataset_path'] + '/en/en.txt'))
	gender_dict.update(parse_gender_dict(options['dataset_path'] + '/es/es.txt'))
	
	# Creating a dict with an incremental id as key and the array [features, author_gender] as value, for each image
	dataset = createDataset(author_images, gender_dict)
	
	# Getting X and Y vectors from the dataset
	X, y, ids = getXandYandIds(dataset)
	
	
	# Converting X to sparse matrix, for each type of feature
	convertLabelToInt = None
	with open(options['convertLabelToInt_path'] , "rb" ) as input_file:
		convertLabelToInt = pickle.load(input_file)
	
	X_objectDetection, convertLabelToInt = X_ToSparseMatrix_objectDetection(X, convertLabelToInt)
	X_colorHistogram, convertLabelToInt = X_ToSparseMatrix_colorHistogram(X)
	X_lbp, convertLabelToInt = X_ToSparseMatrix_lbp(X)
	X_faceRecognition, convertLabelToInt = X_ToSparseMatrix_faceRecognition(X)
	
	# Loading classifiers
	objectDetectionClf = None
	with open(options['clf_path_object_detection'] , "rb" ) as input_file:
		objectDetectionClf = pickle.load(input_file)
	
	faceRecognitionClf = None
	with open(options['clf_path_face_recognition'] , "rb" ) as input_file:
		faceRecognitionClf = pickle.load(input_file)
	
	lbpClf = None
	with open(options['clf_path_lbp'] , "rb" ) as input_file:
		lbpClf = pickle.load(input_file)
	
	colorHistogramClf = None
	with open(options['clf_path_color_histogram'] , "rb" ) as input_file:
		colorHistogramClf = pickle.load(input_file)
	
	# Making the predictions for each classifier
	pred_face_recognition = faceRecognitionClf.predict_proba(X_faceRecognition)
	pred_color_histogram = colorHistogramClf.predict_proba(X_colorHistogram)
	pred_lbp = lbpClf.predict_proba(X_lbp)
	pred_object_detection = objectDetectionClf.predict_proba(X_objectDetection)
	
	# Reconstructing face recognition (for debug)
	datasetPred = dict()
	i=0
	while i < len(pred_face_recognition):
		datasetPred[ids[i]] = {
			'face_recognition': pred_face_recognition[i],
			'color_histogram': pred_color_histogram[i],
			'lbp': pred_lbp[i],
			'object_detection': pred_object_detection[i]
		}
		i+=1
	
	# Building the meta image input data from the output of the classifiers for each entry
	input_meta_image = []
	i=0
	while i < len(pred_face_recognition):
		current_input_entry = [pred_face_recognition[i][0], pred_face_recognition[i][1],
								pred_color_histogram[i][0], pred_color_histogram[i][1],
								pred_lbp[i][0], pred_lbp[i][1],
								pred_object_detection[i][0], pred_object_detection[i][1]]
		input_meta_image.append(current_input_entry)
		i+=1
	
	# Selecting the best classifier from a 20-fold cross validation
	from sklearn.model_selection import cross_val_score
	from sklearn.naive_bayes import MultinomialNB
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.calibration import CalibratedClassifierCV
	from sklearn.svm import LinearSVC
	
	print('Training meta image classifier -- This may take some time..')
	
	classifiers = [MultinomialNB(), RandomForestClassifier(random_state=0, n_estimators=20, n_jobs=-1, max_features=None), DecisionTreeClassifier(random_state=0, max_features=None), LinearSVC(random_state=0)]
	
	best_classifier = None
	best_accuracy = 0
	for clf in classifiers:
		current_scores = cross_val_score(clf, input_meta_image, y, cv=20, scoring='accuracy', n_jobs=-1, verbose=1)
		if current_scores.mean() > best_accuracy:
			best_accuracy = current_scores.mean()
			best_classifier = clf
	
	
	# Training the classifier on the whole data
	if str(type(best_classifier)) == "<class 'sklearn.svm.classes.LinearSVC'>":
		best_classifier = CalibratedClassifierCV(best_classifier)
	best_classifier.fit(input_meta_image, y)
	
	# Saving the classifier
	pickle.dump( best_classifier, open( "trained-classifiers/meta-image-classifier.p", "wb" ) )
	
	print('Best classifier saved for meta image:', best_classifier)
	print('Best accuracy:', best_accuracy)
	
	



if __name__ == "__main__":
	options = {
		'features_path_yolo': "../feature_extractors/extracted-features/author_images_yolo-meta-train.p",
		'features_path_global_features': "../feature_extractors/extracted-features/author_images_global_features-meta-train.p",
		'features_path_face_recognition': "../feature_extractors/extracted-features/author_images_face_recognition-meta-train.p",
		'clf_path_object_detection': "./trained-classifiers/object-detection-classifier.p",
		'clf_path_lbp': "./trained-classifiers/lbp-classifier.p",
		'clf_path_color_histogram': "./trained-classifiers/color-histogram-classifier.p",
		'clf_path_face_recognition': "./trained-classifiers/face-recognition-classifier.p",
		'convertLabelToInt_path': './labels_object_detection/convertLabelToInt.p',
		'dataset_path': "../Min dataset meta image"
	}
	
	train(options)