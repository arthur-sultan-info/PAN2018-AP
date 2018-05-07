import os
import pickle
import re
from time import time

import numpy
from pandas import DataFrame

#------------------------------------------------------------------------------
#---------------------------- UTILITIES MODULE --------------------------------
#------------------------------------------------------------------------------


def save_obj_pickle(path, obj):
    '''
    Save on the disk an object in a compact way
    '''
    with open(path,  'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj_pickle(path):
    '''
    Load from the disk an object saved with "save_obj_pickle" function
    '''
    with open(path, 'rb') as f:
        return pickle.load(f)

def get_printable_tweet(tweet_text):
    '''
    As many utf8 caracters are not convertible to ascii/charmap, this function
    removes unprintable caracters for the console.
    '''
    return re.sub(u'[^\x00-\x7f]',u'', tweet_text)


def build_corpus(authors, label_type, verbosity=1):
    '''
    Given an Author object this function returns a corpus of tweet labelled
    respecting the label type specified
    '''

    if verbosity > 1:
        print("Starting Corpus Building ...")

    # Building tweet Corpus
    t0 = time()
    tweets = []
    labels = []

    for author in authors:
        tweets += author["tweets"]
        labels += [author[label_type] for t in author["tweets"]]

    if verbosity > 1 :
        print("Corpus Building --- success in : " + 
            "{0:.2f}".format(time() - t0) + " seconds" + "\n")


    # At this point, the corpus is a dictionnary with 2 entries:
    # object['tweets'] which contains all the tweets (textual values)
    # object['class'] which contains the classes associated with each tweets
    return {"tweets" : tweets, "labels" : labels}


def get_labels(lang, label_type):
    '''
    Given a configuration of the training (language and label type), returns
    the labels available.
    '''

    if label_type == 'variety':
        return get_variety_labels(lang)

    if label_type == 'gender':
        return ["male", "female"]

    return []


def print_corpus(corpus):
    '''
    Prints all the tweets contained within the corpus give as parameter
    '''
    tweets = corpus['tweets'].values
    for t in tweets: 
        print(get_printable_tweet(t))


def abort_clean (error_msg, error_msg2=""):
    '''
    Stops the execution of the program.
    Displays up to 2 messages before exiting. 
    '''
    print("ERROR : " + error_msg)
    if error_msg2 :
        print("      : " + error_msg2)
    print(" -- ABORTING EXECUTION --")
    print()
    exit()


def format_dir_name(dir_path):
    '''
    Formats the name of the given directory:
        - Transforms to absolute path
        - Ensure there is a '/' at the end
    '''
    path = os.path.abspath(dir_path)
    path = os.path.join(path, '')
    return path


def print_scores(scores):
    '''
    Prints (pretty) the scores object in the console
    '''
    print("Results of the model training :")
    print("    - micro score average: " + str(scores["mean_score_micro"]))
    print("    - macro score average: " + str(scores["mean_score_macro"]))
    print("    - score of the resulting clf: "+str(scores["best_macro_score"]))
    print("    - resulting confusion matrix :")
    
    print(stringify_cm(scores["confusion_matrix"],scores["labels"]))


def stringify_cm(cm, labels, hide_zeroes=False, hide_diagonal=False,
    hide_threshold=None):
    """
    pretty strings for confusion matrixes
    """
    cm_string = ""
    columnwidth = max([len(x) for x in labels]+[10]) # 10 is value length
    empty_cell = " " * columnwidth
    # Print header
    cm_string += "    " + empty_cell + ' '
    for label in labels: 
        cm_string += ("%{0}s".format(columnwidth) % label) + ' '
    cm_string += "\n"
    # Print rows
    for i, label1 in enumerate(labels):
        cm_string += ("    %{0}s".format(columnwidth) % label1) + ' '
        for j in range(len(labels)): 
            cell = "%{0}.0f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            cm_string += cell + ' '
        cm_string += "\n"
    return cm_string


def create_dir(new_dir):
    """
    Checks if the specified direcory does exists
    Creates it if that is not the case
    """
    os.makedirs(new_dir,exist_ok=True)


def get_features_extr_name(feature_union):
    """
    Returns the features extractor name
    """
    name = feature_union[0]
    return name


def get_classifier_name(classifier):
    """
    Returns the classifier name
    """
    return classifier[0]


def integer(string):
    """
    Converts a string to int
    """
    return int(string)


def dir_exists(dir_path):
    """
    Checks if specified directory exists.
    """
    return os.path.isdir(dir_path)


def file_exists(file_path):
    """
    Checks if specified file exists.
    """
    return os.path.isfile(file_path)


def clean_options(args):
    """
    Checks if all options are correct (if all the files/dir they point to exist)
    """

    # input directory - mandatory
    if not(args.input_dir and dir_exists(args.input_dir)):
        abort_clean("Input directory path is incorrect")
    args.input_dir = format_dir_name(args.input_dir)

    # output directory - mandatory
    if not(args.output_dir and dir_exists(args.output_dir)):
        abort_clean("Output directory path is incorrect")
    args.output_dir = format_dir_name(args.output_dir)

    # processed tweets directory - optional
    if args.processed_dir:
        if not(dir_exists(args.processed_tweets_dir)):
            abort_clean("Processed tweets directory path is incorrect")
        else: 
            args.processed_tweets_dir = format_dir_name(
                args.processed_tweets_dir )

    # classifiers directory - optional
    if args.classifiers_dir and not(dir_exists(args.classifiers_dir)):
        abort_clean("Models binaries directory path is incorrect")
    elif args.classifiers_dir: 
        args.classifiers_dir = format_dir_name(args.classifiers_dir)

    # truth directory - optional
    if args.truth_dir and not(dir_exists(args.truth_dir)):
        abort_clean("Truth directory path is incorrect")
    elif args.truth_dir: 
        args.truth_dir = format_dir_name(args.truth_dir)

    # hyper parameters file - optional
    if args.hyper_parameters and not(file_exists(args.hyper_parameters)):
        abort_clean("Hyper parameters file doesn't exist")

    # label type - optional
    if args.label_type:
        if args.label_type == "v" :
            args.label_type = "variety"
        if args.label_type == "g" :
            args.label_type = "gender"
        if not(args.label_type in ["gender", "variety"]) :
            abort_clean("Ill-specified label type")

    # strategy - optional
    if args.aggregation:
        if not(args.aggregation in range(1,101)) :
            abort_clean("Ill-specified strategy")


    return args


def get_language_dir_names():
    '''
    Returns the different language codes available
    [Function specific to PAN18 dataset structure]
    '''
    return ["ar", "en", "es"]


def get_variety_labels(language_code):
    '''
    Returns the different variety labels available for the language code 
    [Function specific to PAN17 dataset structure] 
    '''
    if language_code == "en":
        return ['australia','canada','great britain','ireland',
            'new zealand','united states']
    if language_code == "es":
        return ['argentina','chile','colombia','mexico','peru',
            'spain','venezuela']
    if language_code == "pt":
        return ['portugal','brazil']
    if language_code == "ar":
        return ['gulf','levantine','maghrebi','egypt']
    return []


def get_gender_labels():
    '''
    Returns the different gender labels available for the language code 
    [Function specific to PAN17 dataset structure] 
    '''
    return ["male","female"]