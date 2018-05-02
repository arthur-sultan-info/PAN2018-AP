import cv2
from darkflow.net.build import TFNet
import dataset_parser as parser
from PIL import Image
from IPython.display import clear_output
import os
import argparse

# LOADING PARSER
parser = argparse.ArgumentParser()

parser.add_argument("--dataset", help="Path to the whole dataset")

args = parser.parse_args()

# LOADING YOLO
options = {
    "model": 'cfg/yolo.cfg',
    'load': 'bin/yolo.weights',
    'threshold': 0.2
}

tfnet = TFNet(options)


# EXTRACTING OBJECTS FROM IMAGES OF EACH LANGUAGE
author_images = dict()
i=0
for language_dir in os.listdir(args.dataset):
	directory_path = args.dataset + '/' + language_dir + '/photo'
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

pickle.dump( author_images, open( "author_images_yolo.p", "wb" ) )