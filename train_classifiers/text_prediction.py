import gc
from giovanniScripts.dataset_parser import parse_tweets_from_dir_2
from giovanniScripts.persistance import load_classifiers, save_author_file
from giovanniScripts.utils import abort_clean, format_dir_name
from giovanniScripts.utils import create_dir
from time import time


def classify_authors(Authors, classifiers, outputDic, classification_type='loose', verbosity=1):
    '''
    Return the 'gender_txt' prediction for each user

    Classifies all the tweets contained within a directory.
    Will proceed as follows :
        - predicts the different labels for each author within the corpus
        - returns the most probable labels for each author
    '''

    for auth in Authors:
        # classify gender
        gdr_classes, gdr_predictions = predict_author_proba(
            author=auth,
            model=classifiers["gender"])
        gdr_max_idx = gdr_predictions.index(max(gdr_predictions))
        gdr_predicted = gdr_classes[gdr_max_idx]
        #print(auth["id"] + ":::" + gdr_predicted + "(" + "{0:.2f}".format(gdr_predictions[gdr_max_idx] * 100) + "%)")
        #print(gdr_predictions)
        #print(auth["id"] + ":::{0:.2f}".format(gdr_predictions[0])+":::{0:.2f}".format( gdr_predictions[1]))
        auth["gender_txt"] = gdr_predicted
        outputDic[auth["id"]] = list()
        outputDic[auth["id"]].append("{0:.2f}".format(gdr_predictions[0]))   #female
        outputDic[auth["id"]].append("{0:.2f}".format(gdr_predictions[1]))   #male
    return Authors

def predict_author_proba(author, model):
    '''
    Classify the author object based on the tweets it contains
    Predicts the value of the "meta_label" using the model prediction method
    '''
    predicted_list = []
    #classes = model.classes_.tolist()
    classes = ['female', 'male']
    predictions = [0 for c in classes]

    # Handles the empty file event
    if len(author["tweets"]) == 0:
        predictions[0] = 1
        return classes, predictions

    # it is preferable to use predict_proba (in terms of statistical accuracy)
    # but this method is not always available
    if getattr(model, "predict_proba", None):
        predicted_list = model.predict_proba(author["tweets"])
        for row in predicted_list:
            predictions = [x + y for x, y in zip(predictions, row)]
    else:
        predicted_list = model.predict(author["tweets"])
        for row in predicted_list:
            predictions[classes.index(row)] += 1

    predictions = [x / sum(predictions) for x in predictions]
    return classes, predictions


def predict(inputPath, inputDict, classifierPath, outputPath=None, verbosity_level=1):
    '''

    Given inputPath and inputDict it return outputDic which contains the prediction results

    :param inputPath:  Path to PAN18 dataset
    :param inputDict: { 'ar':[arUser0, .. , arUserN],
                        'en':[enUser0, .. , enUserN]
                        'es':[esUser0, .. , esUserN]}
    :param classifierPath: Path to the dir containing the classifiers produced by 'text_training.py'
    :param outputPath: Path to the dir that will contain the prediction results
    :return outputDic : { userId: [femaleScore, maleScore]}
    '''

    outputDic = {}
    # PAN 18 specifics
    for lang in ['ar', 'en', 'es']:

        if verbosity_level:
            print('---------------------------------------')
            print("Language up for classification: '" + lang + "'\n")

        classifier_dir_path = classifierPath+"/"+ lang
        if outputPath is not None:
            output_dir_path = format_dir_name(outputPath + lang)

        # ----------------------------------------------------------------------
        # Load the tweets features

        Authors = parse_tweets_from_dir_2(
            input_dir=format_dir_name(inputPath+"/"+lang+"/"),
            list_authors= inputDict[lang],
            label=False,
            verbosity_level=verbosity_level)

        if not (Authors):
            abort_clean("Tweets loading failed")

        # ----------------------------------------------------------------------
        # Load the classifiers
        classifiers = load_classifiers(
            classifier_dir_path=classifier_dir_path,
            classification_type='loose',
            verbose=verbosity_level)
        # ----------------------------------------------------------------------
        # Start classification, 'txt', 'img' or 'comb'
        if verbosity_level:
            print("Starting authors classification ...")
            t0 = time()
        classify_authors(Authors, classifiers, outputDic, int(verbosity_level))

        ''' if verbosity_level > 1:
            for auth in Authors:
                print(auth["id"] + ":::txt:::" + auth["gender_txt"])'''

        if verbosity_level:
            print("Classification of '" + lang +
                  "' complete in %.3f seconds" % (time() - t0))
            print('---------------------------------------\n')

        if outputPath is not None:
            create_dir(output_dir_path)
            if(output_dir_path is not None):
                for auth in Authors:
                    save_author_file(
                        author=auth,
                        output_dir=output_dir_path,
                        verbose=verbosity_level > 1
                    )
        # for memory issues, free the classifiers objects
        gc.collect()
    return outputDic





'''if __name__ == "__main__":

    dic = dict()
    dic['ar'] = list()
    dic['en'] = list()
    dic['es'] = list()
    for lang in ['ar', 'en', 'es']:
        truth_file = open("pan18/"+lang+"/"+lang+".txt")
        truth_lines = [x.strip().split(':::') for x in truth_file.readlines()]
        for line in truth_lines:
            dic[lang].append(line[0])
    print(dic)
    outputDic = predict(inputPath= "pan18", inputDict=dic,classifierPath="output_txt_train/",
                        verbosity_level=3)
    #print(outputDic)
    print(len(outputDic))'''

