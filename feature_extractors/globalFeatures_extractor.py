import cv2
from PIL import Image
from IPython.display import clear_output
import os
import argparse
import consts as cst

# FUNCTIONS FOR GLOBAL FEATURES EXTRACTION
import numpy as np
import cv2

# Color histogram
def color_histogram_flattened(image, mask=None):
	chans = cv2.split(image)
	colors = ("b", "g", "r")
	features_ch = []

	# loop over the image channels
	for (chan, color) in zip(chans, colors):
		# create a histogram for the current channel and
		# concatenate the resulting histograms for each
		# channel
		hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
		features_ch.extend(hist)

	return np.array(features_ch).flatten()

# Lbp
# import the necessary packages
from skimage import feature
import numpy as np
 
class LocalBinaryPatterns:
	def __init__(self, numPoints, radius):
		# store the number of points and radius
		self.numPoints = numPoints
		self.radius = radius
 
	def describe(self, image, eps=1e-7):
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
		lbp = feature.local_binary_pattern(image, self.numPoints,
			self.radius, method="uniform")
		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, self.numPoints + 3),
			range=(0, self.numPoints + 2))
 
		# normalize the histogram
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)
 
		# return the histogram of Local Binary Patterns
		return hist

desc = LocalBinaryPatterns(24, 8)

def extractGlobalFeatures(dataset_path):
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

				# Resizing image to compute Global Feature Descriptors
				img = cv2.resize(img, (600,600))

				# Color Histogram
				color_histogram = color_histogram_flattened(img)
				features['color_histogram'] = color_histogram

				# Local Binary Patterns
				imageGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				local_binary_patterns = desc.describe(imageGrey)
				features['local_binary_patterns'] = local_binary_patterns

				current_author_images[current_author_images_index] = features
				
			author_images[author_image_dir] = current_author_images
			
			i += 1
			clear_output()
			print(str(float((i/7500)*100)) + ' %')
			
	return author_images

	
def extract():
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
		

	# EXTRACTING OBJECTS FROM IMAGES OF EACH LANGUAGE
	print('Extracting global features -- This takes some time..')

	author_images = extractGlobalFeatures(dataset_path)

	print(len(author_images))

	# SAVING RESULT WITH PICKLE
	import pickle

	print('Saving global features -- This may take some time..')

	pickle.dump( author_images, open( output_path + '/author_images_global_features-low-train.p', "wb" ) )

	print('Global features extracted and saved at', output_path + '/author_images_global_features-low-train.p')

if __name__ == "__main__":
	extract()