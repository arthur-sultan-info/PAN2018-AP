from sklearn.pipeline import Pipeline


#------------------------------------------------------------------------------
#------------------------ AUTOMATED PIPELINE BUILDER --------------------------
#------------------------------------------------------------------------------

def get_pipeline(features_extr, classifier=None, verbose=1):
    '''
    Builds an execution pipeline from the features extractors and the 
    classifier given as parameter.
    '''

    if(verbose):
        print("Starting building Execution Pipeline ... ")

    # pipeline steps
    steps = []

    # features extractors
    if features_extr:
        if isinstance(features_extr, list):
            steps = features_extr
        else:
            steps = [features_extr]
    
    # classifiers
    if classifier :
        steps += [classifier]

    pipe = Pipeline(steps)
    
    if(verbose):
        print("Execution Pipeline built.\n")

    return pipe