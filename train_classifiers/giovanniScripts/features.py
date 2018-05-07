import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import FeatureUnion
from nltk.tokenize import TweetTokenizer
from nltk.util import ngrams

from giovanniScripts.persistance import load_config
from giovanniScripts.pipeline import get_pipeline
from giovanniScripts.utils import abort_clean


class Ngram_extractor:
    '''
    Class for extracting characters and/or words ngrams

    if min_char_ngrams== 0 and max_char_ngrams==0
        ONLY WORDS ngrams extraction

    if min_word_ngrams== 0 and max_word_ngrams==0
        ONLY CHAR ngrams extraction

    min_*_gram == max_*_gram  MEANS that I want that specific number of grams
    '''

    def __init__(self, min_char_ngrams=3, max_char_ngrams=5, min_word_ngrams=1, max_word_ngrams=2 ):
        '''
        I use as default values the ones of the PAN17 winner paper
        '''
        self.onlyWords = (min_char_ngrams== 0) and (max_char_ngrams==0)
        self.onlyChar = (min_word_ngrams == 0) and (max_word_ngrams == 0)
        if self.onlyWords and self.onlyChar:
            abort_clean("Min and/or Max wrong in ngram configuration file.")
        if min_char_ngrams<0 or max_char_ngrams<0 or min_word_ngrams<0 or max_word_ngrams<0:
            abort_clean("Min and/or Max wrong in ngram configuration file.")
        if max_char_ngrams<min_char_ngrams or max_word_ngrams<0<min_word_ngrams:
            abort_clean("Min and/or Max wrong in ngram configuration file.")

        self.min_char_ngrams = min_char_ngrams
        self.max_char_ngrams = max_char_ngrams
        self.min_word_ngrams = min_word_ngrams
        self.max_word_ngrams = max_word_ngrams

    def get_word_ngrams(self, text, n ):
        tokenizr = TweetTokenizer(preserve_case=True,strip_handles=True,reduce_len=False)
        n_grams = ngrams(tokenizr.tokenize(text), n)
        return [' '.join(grams) for grams in n_grams]

    def get_char_ngrams(self, text, n):
        tokenizr = TweetTokenizer(preserve_case=True,strip_handles=True,reduce_len=False)
        words = tokenizr.tokenize(text)
        res = list()
        for w in words:
            for i in range(len(w)-n+1):
                res.append(w[i:i+n])
        return res

    def analyzer(self, text):
        res = list()
        if not self.onlyChar:
            for i in range(self.min_word_ngrams,self.max_word_ngrams+1):
                res += self.get_word_ngrams(text, i)
        if not self.onlyWords:
            for i in range(self.min_char_ngrams,self.max_char_ngrams+1):
                res += self.get_char_ngrams(text, i)
        return res

    def printArgs(self):
        print(self.min_char_ngrams, self.max_char_ngrams, self.min_word_ngrams, self.max_word_ngrams)


#------------------------------------------------------------------------------
#------------------ AUTOMATED FEATURES EXTRACTORS FETCHER ---------------------
#------------------------------------------------------------------------------

def get_features_extr(features_str_list, verbose=1):
    '''
    Returns a feature union object containing all the features extractor 
    referenced to in the features_str_list.
    '''
    features_str_list = features_str_list.split("+")
    feat_extr_list = []
    # final feature extractor name
    feat_extr_union_name = ""

    if(verbose):
        print("Starting loading features extractor ... ")
    
    # load each features vectorizer and build the union
    # the name of each sub extractor is the final estimator
    for feat_extr_str in features_str_list:
        feat_extr = load_features_extr(feat_extr_str, verbose)
        feat_extr_pipe_name = feat_extr[-1][0]
        feat_extr_pipe = get_pipeline(
            features_extr=feat_extr,
            classifier=None,
            verbose=verbose>2
            )
        feat_extr_list.append((feat_extr_pipe_name,feat_extr_pipe))
        feat_extr_union_name += "+" + feat_extr_pipe_name
        
    feat_extr_union_name = feat_extr_union_name[1:]
    feat_extr_union = FeatureUnion(feat_extr_list)
    res = (feat_extr_union_name, feat_extr_union)
    
    if(verbose):
        print("features extractor loaded : " + feat_extr_union_name + "\n")
    return res


def load_features_extr(features_str, verbose=1):
    '''
    Returns a list of vectorizers to match the specified features_str
    Available features extractors are :
        - wc2   : Word count - bigram
        - char_word_ngrams : char and/or word grams
        - tfidf : TF-IDF
        - lsa   : Latent Semantic Analysis

    A feature extractor can be specified :
        - by its name --> a default clf will be instanciated
        - by a path to a config file, --> a custom clf will be instanciated
    '''
    feat_extractors = []

    #--------------------------------------------------------------------------
    # Get required features_extractor

    if features_str == "wc2":
        feat_extractors.append(get_wc2(None))
    if features_str == "char_word_ngrams":
        feat_extractors.append(get_char_words_ngrams(None))
    elif features_str == "tfidf":
        feat_extractors.append(get_wc2(None))
        feat_extractors.append(get_tfidf(None))
    elif features_str == "tfidfv2":
        feat_extractors.append(get_char_words_ngrams(None))
        feat_extractors.append(get_tfidf(None))
    elif features_str == "lsa":
        feat_extractors.append(get_wc2(None))
        feat_extractors.append(get_tfidf(None))
        feat_extractors.append(get_lsa(None))
    else:
        try: 
            config = load_config(features_str)
        except:
            abort_clean("Cannot load the extractors configuration",
                "Either extr name is incorrect or path is invalid : " +
                features_str)
        # Load the config from a file
        if verbose:
            print("Loading features extractor config from file ")
        feat_extractors = load_features_extr_from_file(config, verbose=verbose)

    #--------------------------------------------------------------------------
    # Return features extractors
    return feat_extractors


def load_features_extr_from_file(config, verbose=None):
    '''
    Returns a list of feature extractors following the given configuration
    '''
    feat_extractors = []
    # get each extractor separately 
    for extr_conf in config["extractors"]:
        if extr_conf["extractr_type"] == "wc2":
            feat_extractors.append(get_wc2(extr_conf))
        elif extr_conf["extractr_type"] == "char_words_ngrams":
            feat_extractors.append(get_char_words_ngrams(extr_conf))
        elif extr_conf["extractr_type"] == "tfidf":
            feat_extractors.append(get_tfidf(extr_conf))
        elif extr_conf["extractr_type"] == "lsa":
            feat_extractors.append(get_lsa(extr_conf))
    return feat_extractors

#------------------------------------------------------------------------------
#--------------------- FEATURES EXTRACTORS CONFIGURATORS ----------------------
#------------------------------------------------------------------------------


# Word Count (unigram and bigram)
#------------------------------------------------------------------------------
def get_wc2(config=None):
    '''
    Returns a word count (bigram) vectorizer.
    If specified, follows the config to setup the vectorizer
    Else follows default wc2 setup.
    '''
    extractr_name = ""
    extractr = None
    tokenizr = TweetTokenizer(
        preserve_case=True,
        strip_handles=True,
        reduce_len=False)

    if not (config):
        extractr_name = "wc2-default"
        extractr = CountVectorizer( #----------------- Default Values
            input='content',
            encoding='utf-8',
            decode_error='ignore',
            strip_accents=None,
            analyzer='word',
            preprocessor=None,
            tokenizer=tokenizr.tokenize, #------------ None
            ngram_range=(1, 2), #--------------------- (1, 1)
            stop_words=None,
            lowercase=True,
            token_pattern=r"(?u)\b\w\w+\b",
            max_df=1.0,
            min_df=2, #------------------------------- 1
            max_features=None,
            vocabulary=None,
            binary=False,
            dtype=np.int64)

    else:
        extractr_name = config["extractr_name"]
        try:
            # Adjustements due to JSON incompatibility
            config["configuration"]["ngram_range"] = tuple(
                config["configuration"]["ngram_range"] )
            config["configuration"]["dtype"] = np.int64
            config["configuration"]["tokenizer"] = tokenizr.tokenize

            extractr = CountVectorizer(**(config["configuration"]))
        except:
            abort_clean("Features Extractor configuration failed",
                "Configuring " + config["extractr_type"] + " with : " + 
                config["configuration"])

    res = (extractr_name, extractr)
    return res

def get_char_words_ngrams(config=None):
    '''
    Returns a vectorizer based on character and/or word ngrams
    If specified, follows the config to setup the vectorizer (min_char_ngrams, max_char_ngrams, min_word_ngram, max_word_ngram)
    Else follows default char_words_ngrams setup (3,5 1,2)
    '''
    extractr_name = ""
    extractr = None
    if not (config):
        extractr_name = "char_words_ngrams"
        min_char_ngrams = 3
        max_char_ngrams = 5
        min_word_ngrams = 1
        max_word_ngrams = 2
        ngram_extractor = Ngram_extractor(int(min_char_ngrams), int(max_char_ngrams), int(min_word_ngrams),
                                          int(max_word_ngrams))
        extractr = CountVectorizer( #----------------- Default Values
            input='content',
            encoding='utf-8',
            decode_error='ignore',
            strip_accents=None,
            analyzer=ngram_extractor.analyzer,
            preprocessor=None,
            tokenizer=None,
            ngram_range=None,
            stop_words=None,
            lowercase=True,
            max_df=1.0,
            min_df=2, #-------------------------------
            max_features=None,
            vocabulary=None,
            binary=False,
            dtype=np.int64)
    else:
        extractr_name = config["extractr_name"]
        try:
            # Adjustements due to JSON incompatibility
            config["configuration"]["dtype"] = np.int64
            config["configuration"]["analyzer"] = "ngram_extractor.analyzer"
            min_char_ngrams = config["configuration"]["min_char_ngrams"]
            max_char_ngrams = config["configuration"]["max_char_ngrams"]
            min_word_ngrams = config["configuration"]["min_word_ngrams"]
            max_word_ngrams = config["configuration"]["max_word_ngrams"]
            ngram_extractor = Ngram_extractor(int(min_char_ngrams), int(max_char_ngrams), int(min_word_ngrams), int(max_word_ngrams))
            ngram_extractor.printArgs()
            extractr = CountVectorizer(  # ----------------- Default Values
                input='content',
                encoding='utf-8',
                decode_error='ignore',
                strip_accents=None,
                analyzer=ngram_extractor.analyzer,
                preprocessor=None,
                tokenizer=None,
                ngram_range=None,
                stop_words=None,
                lowercase=True,
                max_df=1.0,
                min_df=2,  # ------------------------------- 1
                max_features=None,
                vocabulary=None,
                binary=False,
                dtype=np.int64)
        except:
            abort_clean("Features Extractor configuration failed",
                        "Configuring " + config["extractr_type"] + " with : " +
                        config["configuration"])

    res = (extractr_name, extractr)
    return res


# Term Frequency - Inverse Document Frequency
#------------------------------------------------------------------------------
def get_tfidf(config=None):
    '''
    Returns a tfidf vectorizer.
    If specified, follows the config to setup the vectorizer
    Else follows default tfidf setup.
    '''
    extractr_name = ""
    extractr = None

    if not (config):
        extractr_name = "tfidf-default"
        extractr = TfidfTransformer(
            norm='l2',
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=False)

    else:
        extractr_name = config["extractr_name"]
        try:
            #extract parameter from config json file
            use_idf = config["configuration"]["use_idf"]
            smooth_idf = config["configuration"]["smooth_idf"]
            sublinear_tf =  config["configuration"]["sublinear_tf"]
            print(use_idf, smooth_idf, sublinear_tf)
            extractr = TfidfTransformer(
                norm='l2',
                use_idf=bool(use_idf),
                smooth_idf=bool(smooth_idf),
                sublinear_tf=bool(sublinear_tf))
        except:
            abort_clean("Features Extractor configuration failed",
                "Configuring " + config["extractr_type"] + " with : " + 
                config["configuration"])

    res = (extractr_name, extractr)
    return res

# Latent Semantic Analysis
#------------------------------------------------------------------------------
def get_lsa(config=None):
    '''
    Returns a latent semantic analysis vectorizer.
    If specified, follows the config to setup the vectorizer
    Else follows default lsa setup.
    '''
    extractr_name = ""
    extractr = None

    if not (config):
        extractr_name = "lsa-default"
        extractr = TruncatedSVD( #------------------------- Default Values
            n_components=1000, #---------------------- 2
            algorithm="randomized",
            n_iter=10,
            random_state=42,
            tol=0.
        )

    else:
        extractr_name = config["extractr_name"]
        try:
            extractr = TruncatedSVD(**(config["configuration"]))
        except:
            abort_clean("Features Extractor configuration failed",
                "Configuring " + config["extractr_type"] + " with : " +
                config["configuration"])

    res = (extractr_name, extractr)
    return res


# Doc2Vec
#------------------------------------------------------------------------------
def get_doc2vec(conf, dm, verbose=False):
    '''
    Instanciate a gensim doc given a conf file path.
    Returns the doc2vec model configured.
    If the conf file path is not specified, returns a default doc2vec model.
    '''

    if conf:
        vector_size = conf["configuration"]["vector_size"]
        window      = conf["configuration"]["window"]
        min_count   = conf["configuration"]["min_count"]
    else:
        vector_size = 300
        window = 5
        min_count = 2
    
    # import gensim models (heavy load)
    from gensim import models as gensim_models
    # creates a new doc2vec model
    model = gensim_models.Doc2Vec(
        documents=None, 
        dm_mean=None, 
        dm=dm,
        dbow_words=0, 
        dm_concat=0,
        dm_tag_count=1, 
        docvecs=None, 
        docvecs_mapfile=None, 
        comment=None, 
        trim_rule=None,
        size=vector_size, 
        window=window, 
        min_count=min_count, 
        workers=8,
        alpha=0.025, 
        min_alpha=0.0025 )
    
    return model
