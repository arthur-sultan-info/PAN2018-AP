from os.path import join
from sklearn.model_selection import train_test_split
from random import uniform
from pickle import dump, HIGHEST_PROTOCOL

def splitting(inputPath, outputPath):
    '''

    For each language, splits in balanced way the dataset in  partitions of 80% and 20%
    The output dictionaries are saved as .pkl files in the outputPath directory.

    :param inputPath:  Path to PAN18 dataset
    :param outputPath: Path to the directory that will contain the .pkl files,
        NB. Create outputPath directory before using this function
    '''
    for lang in ['ar','en','es']:
        input_dir = inputPath + '/' + lang
        output_dir = join(outputPath)
        with open(input_dir+"/"+lang+".txt") as truth_file:
            truth_lines = [x.strip().split(':::') for x in truth_file.readlines()]

            d = dict()
            d['female'] = list()
            d['male'] = list()
            for line in truth_lines :
                d[line[1]].append(line[0])

            split1, split2 = train_test_split(d['male'], test_size=0.2, random_state=int(uniform(0, 1)*100))
            split3, split4 = train_test_split(d['female'], test_size=0.2, random_state=int(uniform(0, 1) * 100))

            d = dict()
            for userid in split1:
                d[userid] = 0
            for userid in split3:
                d[userid] = 0
            for userid in split2:
                d[userid] = 1
            for userid in split4:
                d[userid] = 1

            with open(output_dir+"/"+lang+".pkl", 'wb') as f:
                dump(d, f, HIGHEST_PROTOCOL)


if __name__ == "__main__":
    splitting("PAN dataset/pan18-author-profiling-training-2018-02-27","output/splitting")