from os.path import join
from sklearn.model_selection import train_test_split
from random import uniform
import pickle
import os

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

def splitting(inputPath, outputPath):
	options = {
		'dataset_path': "PAN dataset/pan18-author-profiling-training-2018-02-27"
	}
	
	# Parsing gender
	gender_dict = parse_gender_dict(options['dataset_path'] + '/ar/ar.txt')
	gender_dict.update(parse_gender_dict(options['dataset_path'] + '/en/en.txt'))
	gender_dict.update(parse_gender_dict(options['dataset_path'] + '/es/es.txt'))
	
	# Splitting each language
	splitDic = dict()
	splitDic80 = dict()
	split70 = dict()
	split20 = dict()
	split10 = dict()
	
	for lang in ['ar','en','es']:
		with open(inputPath + '/' + lang + '.pkl', 'rb') as input_file:
			splitDic = pickle.load(input_file)
	
		splitDic80[lang] = dict()
		for author in splitDic:
			if splitDic[author] == 0:
				splitDic80[lang][author] = gender_dict[author]
		
		cptDispatchMale = 1
		cptDispatchFemale = 1
		split70[lang] = dict()
		split20[lang] = dict()
		split10[lang] = dict()
		for author in splitDic80[lang]:
			currentAuthorGender = splitDic80[lang][author]
			if currentAuthorGender == 0:					
				if cptDispatchFemale < 3:
					split20[lang][author] = currentAuthorGender
					cptDispatchFemale += 1
				elif cptDispatchFemale < 10:
					split70[lang][author] = currentAuthorGender
					cptDispatchFemale += 1
				else:
					split10[lang][author] = currentAuthorGender
					cptDispatchFemale += 1
					cptDispatchFemale = 1
				
			else:
				if cptDispatchMale < 3:
					split20[lang][author] = currentAuthorGender
					cptDispatchMale += 1
				elif cptDispatchMale < 10:
					split70[lang][author] = currentAuthorGender
					cptDispatchMale += 1
				else:
					split10[lang][author] = currentAuthorGender
					cptDispatchMale += 1
					cptDispatchMale = 1
		
	# Joining splits in one big dict
	split70_20_10 = dict()
	for lang in split70:
		split70_20_10[lang] = dict()
		for author in split70[lang]:
			split70_20_10[lang][author] = 70
		for author in split20[lang]:
			split70_20_10[lang][author] = 20
		for author in split10[lang]:
			split70_20_10[lang][author] = 10
	
	# Getting each lang in one dic for saving in one file for each language
	split70_20_10_ar = split70_20_10['ar']
	split70_20_10_en = split70_20_10['en']
	split70_20_10_es = split70_20_10['es']
	
	# Saving splits
	pickle.dump( split70_20_10_ar, open( outputPath + '/ar.pkl', "wb" ) )
	pickle.dump( split70_20_10_en, open( outputPath + '/en.pkl', "wb" ) )
	pickle.dump( split70_20_10_es, open( outputPath + '/es.pkl', "wb" ) )
	

if __name__ == "__main__":
    splitting("output/splitting-image-text","output/splitting-low-meta-combine")