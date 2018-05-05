import cv2
from darkflow.net.build import TFNet
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


# LOADING YOLO
options = {
    "model": 'cfg/yolo.cfg',
    'load': 'bin/yolo.weights',
    'threshold': 0.2,
	'gpu': 1.0
}

tfnet = TFNet(options)


# EXTRACTING OBJECTS FROM IMAGES OF EACH LANGUAGE
print('Extracting object detection features -- This takes some time..')

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
			
			# YOLO labels
			result = tfnet.return_predict(img)
			
			labels = dict()
			z=0
			while(z < len(result)):
				if(result[z]['label'] not in labels):
					labels[result[z]['label']] = result[z]['confidence']
				else:
					labels[result[z]['label']] += result[z]['confidence']
				z+=1
			
			features['labels'] = labels
			
			current_author_images[current_author_images_index] = features
		author_images[author_image_dir] = current_author_images
		
		i += 1
		clear_output()
		print(str(float((i/7500)*100)) + ' %')

print(len(author_images))

# SAVING RESULT WITH PICKLE
import pickle

print('Saving object detection features -- This may take some time..')

pickle.dump( author_images, open( output_path + '/author_images_yolo-low-train.p', "wb" ) )

print('Object detection features extracted and saved at', output_path + '/author_images_yolo-low-train.p')