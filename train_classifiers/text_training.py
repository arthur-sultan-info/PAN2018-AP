from os.path import join, exists
from os import makedirs
from pickle import load
from time import time
from numpy import array
from shutil import rmtree

from giovanniScripts.dataset_parser import parse_tweets_from_dir
from utils import abort_clean


from giovanniScripts.classifiers import get_classifier
from giovanniScripts.features import get_features_extr
from giovanniScripts.persistance import save_scores, save_model
from giovanniScripts.pipeline import get_pipeline
from giovanniScripts.utils import build_corpus, abort_clean, print_scores
from giovanniScripts.utils import get_classifier_name, get_features_extr_name, get_labels

from sklearn.base import clone
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import KFold

def predict_author_proba(author, model):
    '''
    Classify the author object based on the tweets it contains
    Predicts the value of the "meta_label" using the model prediction method
    '''
    predicted_list = []
    classes = model.classes_.tolist()
    #classes = ['female', 'male']
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

    predictions = [x/sum(predictions) for x in predictions]
    return classes, predictions


def train_model(corpus, pipeline, verbose):
    '''
    Takes a pipeline and train it on the specified corpus.
    Returns the trained pipeline once finished.
    '''

    if verbose:
        t0 = time()
        print("Starting model training ... (this may take some time)")

    # retrieve tweets and labels
    train_tweets = corpus['tweets']
    train_labels = corpus['labels']

    # train the pipeline
    pipeline.fit(train_tweets, train_labels)
    if verbose:
        print("Model training complete in %.3f seconds\n" % (time() - t0))

    return pipeline


def train_model_cross_validation(authors, label_type, pipeline, verbose=1):
    '''
    Takes a pipeline and train it on the specified corpus.
    Processes a cross-validation algorithm (K-fold) in order to evaluate the
    quality of the model.
    Returns the best trained pipeline (in terms of macro f-score).
    '''

    labels = get_labels(
        lang=authors[0]["lang"],
        label_type=label_type)

    if not (labels):
        abort_clean("Could not extract labels")
    if verbose:
        print("Labels extraction succeded.")
        print("Available labels : " + " / ".join(labels) + "\n")

    if verbose:
        t0 = time()
        print("Starting model Cross Validation ... (this may take some time)")

    confusion = array(
        [[0 for x in range(len(labels))] for y in range(len(labels))])
    scores = []
    best_f_score = 0
    best_pipeline = None
    scores_micro = []
    scores_macro = []

    # start Kfold cross validation.
    n_run = 1
    k_fold = KFold(n_splits=10, shuffle=True)
    authors = array(authors)
    for train_indices, test_indices in k_fold.split(authors):

        #print("train_indices+ ",train_indices)
        # build train corpus
        train_authors = authors[train_indices]
        train_corpus = build_corpus(
            authors=train_authors,
            label_type=label_type,
            verbosity=verbose)

        # build test corpus
        test_authors = authors[test_indices]

        # train model
        pipeline = train_model(
            corpus=train_corpus,
            pipeline=pipeline,
            verbose=0)

        # test model
        truthes = []
        predictions = []
        for author in test_authors:
            #pipeline.fit(author['tweets'])
            var_classes, var_predictions = predict_author_proba(
                author=author,
                model=pipeline)
            var_max_idx = var_predictions.index(max(var_predictions))
            label_predicted = var_classes[var_max_idx]
            predictions.append(label_predicted)
            truthes.append(author[label_type])

        # compute metrics
        confusion += confusion_matrix(truthes, predictions, labels=labels)
        score_micro = f1_score(truthes, predictions,
                               labels=labels, average="micro")
        score_macro = f1_score(truthes, predictions,
                               labels=labels, average="macro")

        if verbose:
            print("Fold " + str(n_run) + " : micro_f1=" + str(score_micro) +
                  " macrof1=" + str(score_macro))

        # store for avg
        scores_micro.append(score_micro)
        scores_macro.append(score_macro)
        n_run += 1

        # save the pipeline if better than the current one
        if score_macro > best_f_score:
            best_pipeline = clone(pipeline, True)
            best_f_score = score_macro
            best_train_indices = train_indices
            best_test_indices = test_indices

    if verbose:
        print("Model Cross Validation complete in %.3f seconds.\n"
              % (time() - t0))

    scores = {"mean_score_micro": sum(scores_micro) / len(scores_micro),
              "mean_score_macro": sum(scores_macro) / len(scores_macro),
              "confusion_matrix": confusion,
              "best_macro_score": best_f_score,
              "labels": labels}

    return best_pipeline, scores, best_train_indices, best_test_indices


def train(inputPath, splitsPath,  outputPath, verbosity_level=1):
    '''

    For each language, proceeds as follow:
        - takes in input the corresponding .pkl file
        - train a text-based classifier on the 80% split
        - save the resulting model in outputPath

    :param inputPath:  Path to PAN18 dataset
    :param splitsPath: Path to dir containing the .pkl files produced by 'splitting.py'
    :param outputPath: Path to dir in which the outputs models will be saved
        NB. Create outputPath directory before using this function
    '''

    for lang in ['ar', 'en', 'es']:

        input_dir = join(inputPath, lang)
        output_dir = join(outputPath, lang)

        #print("input_dir ", input_dir)
        #print("output_dir ", output_dir)

        if exists(output_dir):
            rmtree(output_dir)
        makedirs(output_dir)

        # --------------------------------------------------------------------------
        # Load the .pkl file
        with open(splitsPath + "/" + lang + ".pkl", 'rb') as f:
            dic = load(f)
        # Load the tweets in one language
        Authors = parse_tweets_from_dir(
            input_dir=inputPath+"/"+lang+"/",
            label=True,
            aggregation=100,
            splitDic=dic,
            verbosity_level=verbosity_level)

        if not (Authors):
            abort_clean("Tweets loading failed")

        # --------------------------------------------------------------------------
        # Load the classifier

        t0 = time()
        classifier = get_classifier(
            classifier_str="svm",
            config=None,
            verbose=verbosity_level)

        # --------------------------------------------------------------------------
        # Load the features extractors

        features_extr = None
        features_extr = get_features_extr(
                features_str_list="tfidf",
                verbose=verbosity_level)
        # --------------------------------------------------------------------------
        # Build the execution pipeline

        pipeline = get_pipeline(
            features_extr=features_extr,
            classifier=classifier,
            verbose=verbosity_level)

        # --------------------------------------------------------------------------
        # Train the execution pipeline

        # train and cross validate results
        if (verbosity_level):
            print("Model Training with cross validation\n")
        pipeline, scores, best_train_indices, best_test_indices = train_model_cross_validation(
            authors=Authors,
            label_type="gender",
            pipeline=pipeline,
            verbose=verbosity_level)

        if verbosity_level:
            print_scores(scores)

        filename = str(get_features_extr_name(features_extr) +
                       "+" + get_classifier_name(classifier))

        save_scores(
            scores=scores,
            output_dir=output_dir+"/",
            filename=lang,
            verbose=verbosity_level)

        #--------------------------------------------------------------------------
        # Save the resulting model
        filename = str(get_features_extr_name(features_extr) +
                           "+" + get_classifier_name(classifier))

        # build train corpus
        authors = array(Authors)
        train_authors = authors[best_train_indices]
        train_corpus = build_corpus(
            authors=train_authors,
            label_type='gender',
            verbosity=verbosity_level)
        # build test corpus
        test_authors = authors[best_test_indices]

        # train model
        pipeline = train_model(
            corpus=train_corpus,
            pipeline=pipeline,
            verbose=0)

        save_model(
                pipeline=pipeline,
                output_dir=output_dir+"/",
                filename=filename,
                verbose=verbosity_level)

        # --------------------------------------------------------------------------
        # End Execution
        if verbosity_level:
            print("Training task complete in " + str(round(time() - t0)) + " s")



'''if __name__ == "__main__":
    train(inputPath="pan18",splitsPath="splits",outputPath="output_txt_train",verbosity_level=3)'''