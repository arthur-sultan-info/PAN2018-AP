import cv2
from PIL import Image
from IPython.display import clear_output
import os
import argparse
import consts as cst

# LOADING PARSER
parser = argparse.ArgumentParser()

parser.add_argument("--dataset", help="Path to the whole dataset")
parser.add_argument("--output", help="Path to the output file")

args = parser.parse_args()

if(args.dataset is None):
	dataset_path = cst.DEFAULT_DATASET_PATH
else:
	dataset_path = args.dataset
	
if(args.output is None):
	output_path = cst.DEFAULT_OUTPUT_PATH
else:
	output_path = args.output

# LOADING FACE RECOGNITION
from face_recognition import face_locations
from PIL import Image, ImageDraw, ImageFont
from faceClassification import pred
from tqdm import tqdm
import pickle

with open('faceClassification/face_model.pkl', 'rb') as f:
        clf, labels = pickle.load(f)



# EXTRACTING FACES FROM IMAGES OF EACH LANGUAGE
print('Extracting face detection features -- This takes some time..')

author_images = dict()
i=0
for language_dir in os.listdir(dataset_path):
	directory_path = dataset_path + '/' + language_dir + '/photo'
	print('Extracting object detection features for language', language_dir)
	for author_image_dir in os.listdir(directory_path):    
		current_author_images = dict()
		current_author_images_index = -1
		for filename in os.listdir(directory_path + '/' + author_image_dir):
			current_author_images_index += 1
			if(filename.split('.')[2] != 'jpeg'):
				continue
			img = cv2.imread(directory_path + '/' + author_image_dir + '/' + filename, cv2.IMREAD_COLOR)
			if(img is None):
				continue
				
			features = dict()
			
			numberOfMalesRecognized = 0
			numberOfFemalesRecognized = 0
			prediction, locs = pred.predict_one_image(directory_path + '/' + author_image_dir + '/' + filename, clf, labels)
			
			if(prediction is not None):
				prediction = prediction.to_dict()
				for recognizedPersonIndex in prediction['Male']:
					if prediction['Male'][recognizedPersonIndex] >= 0.5:
						numberOfMalesRecognized += 1
					else:
						numberOfFemalesRecognized += 1

			features['Male'] = numberOfMalesRecognized
			features['Female'] = numberOfFemalesRecognized
			
			current_author_images[current_author_images_index] = features
		author_images[author_image_dir] = current_author_images
		
		i += 1
		clear_output()
		print(str(float((i/250)*100)) + ' %')

print(len(author_images))

# SAVING RESULT WITH PICKLE

print('Saving face detection features -- This may take some time..')

pickle.dump( author_images, open( output_path + '/author_images_face_recognition-low-train.p', "wb" ) )

print('Face detection features extracted and saved at', output_path + '/author_images_face_recognition-low-train.p')