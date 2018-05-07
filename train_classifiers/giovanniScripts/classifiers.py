from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from giovanniScripts.persistance import load_config
from giovanniScripts.utils import abort_clean


#------------------------------------------------------------------------------
#----------------------- AUTOMATED CLASSIFIERS FETCHER ------------------------
#------------------------------------------------------------------------------

def get_classifier(classifier_str, config=None, verbose=1):
    '''
    Returns a classifier specified in parameter
    Available classifiers are :
        - nbb : NaiveBayes (bernouilly)
        - mlp : Multi-layered Perceptron
        - rfo : Random Forest
        - svm : Support Vector Machine

    A classifier can be specified : (TODO)
        - by its name --> a default ft_extr will be instanciated
        - by a path to a config file, --> a custom ft_extr will be instanciated
    '''

    if verbose and not(config):
        print("Starting loading classifier ... ")
    if config:
        classifier_str = config["classifier_type"]
    
    #--------------------------------------------------------------------------
    # Get required classifier

    clf_name = ""
    clf = None

    if classifier_str == "svm":
        clf_name, clf = get_svm(config)
    
    elif classifier_str == "mlp":
        clf_name, clf = get_mlp(config)

    elif classifier_str == "nbb":
        clf_name, clf = get_nbb(config)

    elif classifier_str == "rfo":
        clf_name, clf = get_rfo(config)

    else:
        try: 
            config = load_config(classifier_str)
        except:
            abort_clean("Cannot load the classifier configuration",
                "Either the clf name is incorrect or the path is invalid : " +
                classifier_str)

        if verbose:
            print("Loading classifier config from file")
        # recursive call with config loaded
        return get_classifier("", config, verbose=verbose)


    
    #--------------------------------------------------------------------------
    # Return classifier
    if(verbose):
        print("classifier loaded: '" + clf_name + "'\n")

    res = (clf_name, clf)
    return res


#------------------------------------------------------------------------------
#------------------------- CLASSIFIERS CONFIGURATORS --------------------------
#------------------------------------------------------------------------------


# Support Vector Machine
#------------------------------------------------------------------------------
def get_svm(config=None):
    '''
    Returns a svm classifier.
    If specified, follows the config to setup the svm
    Else follows default svm setup.
    '''
    clf_name = ""
    clf = None
    from sklearn.calibration import CalibratedClassifierCV
    if not(config):
        clf_name = "svm-default"
        clf = CalibratedClassifierCV(LinearSVC( #---------------------------- Default Value
                    C=1.0,
                    loss='squared_hinge',
                    penalty='l1', #------------------- l2
                    dual=False, #--------------------- True
                    tol=1e-4,
                    multi_class='crammer_singer', #--- ovr
                    fit_intercept=True,
                    intercept_scaling=1,
                    class_weight=None,
                    verbose=0,
                    random_state=None,
                    max_iter=500)) #-------------------- 1000

    else:
        clf_name = config["classifier_name"]
        try:
            clf = CalibratedClassifierCV(LinearSVC(**(config["configuration"])))
        except:
            abort_clean("Classifier configuration failed",
                "Configuring " + config["classifier_type"] + " with : " + 
                config["configuration"])
        
    return clf_name, clf


# Multi Layered Perceptron
#------------------------------------------------------------------------------
def get_mlp(config=None):
    '''
    Returns a Multi-Layered Perceptron classifier.
    If specified, follows the config to setup the mlp classifier
    Else follows default mlp classifier setup.
    '''
    clf_name = ""
    clf = None

    if not (config):
        clf_name = "mlp-default"
        clf = MLPClassifier(
            hidden_layer_sizes=(100,),
            activation="relu",
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate="constant",
            learning_rate_init=0.001,
            power_t=0.5,
            max_iter=200,
            shuffle=True,
            random_state=None,
            tol=1e-4,
            verbose=False,
            warm_start=False,
            momentum=0.9,
            nesterovs_momentum=True,
            early_stopping=False,
            validation_fraction=0.1,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8)

    else:
        clf_name = config["classifier_name"]
        try:
            config["configuration"]["hidden_layer_sizes"] = tuple(
                config["configuration"]["hidden_layer_sizes"] )
            clf = MLPClassifier(**(config["configuration"]))
        except:
            abort_clean("Classifier configuration failed",
                "Configuring " + config["classifier_type"] + " with : " + 
                config["configuration"])

    return clf_name, clf


# Naive Bayes (Bernouilly)
#------------------------------------------------------------------------------
def get_nbb(config=None):
    '''
    Returns a Naive Bayes classifier (bernouilly implementation).
    If specified, follows the config to setup the NB classifier
    Else follows default NB classifier setup.
    '''
    clf_name = ""
    clf = None

    if not (config):
        clf_name = "nbb-default"
        clf = BernoulliNB(
            alpha=1.0,
            binarize=.0,
            fit_prior=True,
            class_prior=None)

    else:
        clf_name = config["classifier_name"]
        try:
            clf = BernoulliNB(**(config["configuration"]))
        except:
            abort_clean("Classifier configuration failed",
                "Configuring " + config["classifier_type"] + " with : " + 
                config["configuration"])
        
    return clf_name, clf


# Random Forrest
#------------------------------------------------------------------------------
def get_rfo(config=None):
    '''
    Returns a Naive Bayes classifier (bernouilly implementation).
    If specified, follows the config to setup the NB classifier
    Else follows default NB classifier setup.
    '''
    clf_name = ""
    clf = None

    if not (config):
        clf_name = "rfo-default"
        clf = RandomForestClassifier(
            n_estimators=10,
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.,
            max_features="auto",
            max_leaf_nodes=None,
            min_impurity_split=1e-7,
            bootstrap=True,
            oob_score=False,
            n_jobs=-1, #------------------------------ 1
            random_state=None,
            verbose=0,
            warm_start=False,
            class_weight=None)

    else:
        clf_name = config["classifier_name"]
        try:
            clf = RandomForestClassifier(**(config["configuration"]))
        except:
            abort_clean("Classifier configuration failed",
                "Configuring " + config["classifier_type"] + " with : " + 
                config["configuration"])
        
    return clf_name, clf

