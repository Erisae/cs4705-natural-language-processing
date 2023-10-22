#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk import word_tokenize

import numpy as np
import tensorflow

import gensim
import string
import transformers 

from typing import List

def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos) -> List[str]:
    # Part 1
    """
    1. look up the lemma and pos in WordNet
    2. retrieve all synsets that the lemma appears in
    3. obtain all lemmas that appear in any of these synsets
    warning: output does not contain the input lemma itself
    Lemma object represents a word form and its synset in WordNet
    """
    res = set()
    for ll in wn.lemmas(lemma, pos=pos): # retrieve all lexemes for the [pos] [lemma]
        for l in ll.synset().lemmas(): # get the synset for the every lexeme, get all lexemes in that synset
            res.add(l.name().replace("_", " ")) # get word of lexeme
    res.discard(lemma)
    return res

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context : Context) -> str:
    # Part 2
    """
    1. sum up the occurence counts for all senses of the word if the word 
    and the target appear together in multiple synsets
    """

    res = {}
    for lemma in wn.lemmas(context.lemma, pos=context.pos):
        for l in lemma.synset().lemmas():
            key = l.name().replace("_", " ")
            if not key==context.lemma: # filter out target
                if not key in res:
                    res[key] = 0
                res[key] += l.count() # occurence frequency of this sense of s.word() in corpus

    sorted_res = sorted(res.items(), key=lambda x: x[1], reverse=True)
    return sorted_res[0][0]

def wn_simple_lesk_predictor(context : Context) -> str:
    # Part 3
    """
    1. Look at all possible synsets that the target word appears in
    2. Compute the overlap between the definition of the synset and the context of the target word
    3. Select a synset for the target word, return the most frequent synonym from that synset
    """
    target_word = context.lemma
    target_pos = context.pos
    stop_words = set(stopwords.words('english'))

    # construct context
    left  = " ".join(context.left_context)
    right = " ".join(context.right_context)
    sentence = "{left} {right}".format(left=left, right=right)
    context_words = set(tokenize(sentence.lower())).difference(stop_words)

    best_sense = None
    max_score = -1

    # for sense in wn.synsets(target_word):
    for l in wn.lemmas(target_word, pos=target_pos):
        sense = l.synset()
    
        # defination
        def_t = tokenize(sense.definition().lower())

        # exmaples
        exm_t = []
        for example in sense.examples():
            exm_t += tokenize(example.lower())
        
        # hypernyms's defination and examples
        def_ht = []
        exm_ht = []
        for hypernym in sense.hypernyms():
            def_ht += tokenize(hypernym.definition().lower())
            for example in hypernym.examples():
                exm_ht += tokenize(example.lower())

        # filter end words
        signature = set(def_t + exm_t + def_ht + exm_ht).difference(stop_words)
        
        # compute overlap
        overlap = len(signature.intersection(context_words))

        for ll in sense.lemmas():
            if not ll.name() == target_word:
                score = 1000*overlap + 100*l.count() + ll.count()
                if score > max_score:
                    max_score = score
                    best_sense = ll

    if max_score == 0:
        return 'smurf'
    
    return best_sense.name().replace("_", " ")

class Word2VecSubst(object):
    # part 4
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context : Context) -> str:
        # part4
        # first obtain a set of possible synonyms from WordNet [part1]
        target_word = context.lemma
        target_pos = context.pos
        candidates = get_candidates(target_word, target_pos)

        # return the synonym that is most similar to the target word, according to the Word2Vec embeddings
        max_sim = 0
        synonym = None
        for c in candidates:
            try:
                sim = self.model.similarity(c, target_word)
            except KeyError:
                continue
            if sim > max_sim:
                max_sim = sim
                synonym = c
        
        return synonym



class BertPredictor(object):
    # part 5
    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        # part 5
        # obtain a set of candidate synonyms
        candidates = get_candidates(context.lemma, context.pos)

        # masked input representation
        left  = " ".join(context.left_context).lower()
        right = " ".join(context.right_context).lower()
        sentence = "{left} [MASK] {right}".format(left=left, right=right)

        # run DistilBERT model on input representation
        input_toks = self.tokenizer.encode(sentence)
        
        tokens = self.tokenizer.convert_ids_to_tokens(input_toks)
        mask_idx = tokens.index("[MASK]") # store the position of the [MASK] token

        input_mat = np.array(input_toks).reshape((1,-1))
        outputs = self.model.predict(input_mat, verbose = None)
        predictions = outputs[0]
        best_words_idx = np.argsort(predictions[0][mask_idx])[::-1] # Sort in increasing order
        best_words = self.tokenizer.convert_ids_to_tokens(best_words_idx)

        #  Select from [candidate synonyms], the [highest-scoring] word in the target position
        res = None
        for word in best_words:
            if word in candidates:
                res = word
                break
        return res
    

class BertWord2VecSubst(object):
    # part 6 version 1
    """
    combine the score of bert and word2vec
    """
    def __init__(self, filename):
        self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)  
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.bert_model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
        self.all_tokens = self.tokenizer.convert_ids_to_tokens(range(30522))

    def predict(self, context : Context) -> str:
        candidates = list(get_candidates(context.lemma, context.pos))

        # bert
        left  = " ".join(context.left_context).lower()
        right = " ".join(context.right_context).lower()
        sentence = "{left} [MASK] {right}".format(left=left, right=right)
        input_toks = self.tokenizer.encode(sentence)
        
        tokens = self.tokenizer.convert_ids_to_tokens(input_toks)
        mask_idx = tokens.index("[MASK]") # store the position of the [MASK] token

        input_mat = np.array(input_toks).reshape((1,-1))
        outputs = self.bert_model.predict(input_mat, verbose = None)
        predictions = outputs[0]
        
        max_score = -10000
        res = None
        for i in range(len(candidates)):
            try:
                idx = self.all_tokens.index(candidates[i])
                sim = self.word2vec_model.similarity(candidates[i], context.lemma)
                score = predictions[0,mask_idx,idx] + sim

                if score > max_score:
                    max_score = score
                    res = candidates[i]
            except KeyError:
                continue
            except ValueError:
                continue

        return res
    
class LeskWord2VecSubst(object):
    # part 6 version 2
    """
    combine the score of part3 and part4
    """
    def __init__(self, filename):
        self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True) 

    def normalize_dict_values(self, d):
        values = list(d.values())
        min_val = min(values)
        max_val = max(values)

        if sum(values) == 0:
            return d
        
        if min_val == max_val:
            min_val = 0
            max_val = sum(values)
        
        for key in d:
            d[key] = (d[key] - min_val) / (max_val - min_val)
        return d

    def predict(self, context : Context) -> str:

        lesk = {}
        sims = {}

        stop_words = set(stopwords.words('english'))
        left  = " ".join(context.left_context)
        right = " ".join(context.right_context)
        sentence = "{left} {right}".format(left=left, right=right)
        context_words = set(tokenize(sentence.lower())).difference(stop_words)

        for lemma in wn.lemmas(context.lemma, pos=context.pos):
            sense = lemma.synset()
            def_ht = []
            exm_ht = []
            exm_t = []
            def_t = tokenize(sense.definition().lower())
            for example in sense.examples():
                exm_t += tokenize(example.lower())
            for hypernym in sense.hypernyms():
                def_ht += tokenize(hypernym.definition().lower())
                for example in hypernym.examples():
                    exm_ht += tokenize(example.lower())
            signature = set(def_t + exm_t + def_ht + exm_ht).difference(stop_words)
            overlap = len(signature.intersection(context_words))

            for l in lemma.synset().lemmas():
                key = l.name().replace("_", " ")
                if not key==context.lemma: # filter out target
                    if not key in lesk:
                        lesk[key] = 0
                    lesk[key] += 1000*overlap + 100*lemma.count() + l.count()

        for key in lesk.keys():
            try:
                sims[key] = self.word2vec_model.similarity(key, context.lemma)
            except KeyError:
                continue
        
        lesk = self.normalize_dict_values(lesk)
        sims = self.normalize_dict_values(sims)
        res = {key: lesk.get(key, 0)*0.2 + sims.get(key, 0) for key in set(lesk) | set(sims)}

        return max(res, key=lambda k: res[k])


    

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    W2VMODEL_FILENAME = './GoogleNews-vectors-negative300.bin.gz'
    #predictor = Word2VecSubst(W2VMODEL_FILENAME)

    # for context in read_lexsub_xml(sys.argv[1]):
    #     #print(context)  # useful for debugging
    #     prediction = smurf_predictor(context) 
    #     print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))

    ## part1
    # print(get_candidates('slow','a'))

    ## part2
    # for context in read_lexsub_xml(sys.argv[1]):
    #     #print(context)  # useful for debugging
    #     prediction = wn_frequency_predictor(context) 
    #     print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))

    ## part3
    # for context in read_lexsub_xml(sys.argv[1]):
    #     # print(context)  # useful for debugging
    #     prediction = wn_simple_lesk_predictor(context) 
    #     print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))

    # part4
    # word2vec = Word2VecSubst(W2VMODEL_FILENAME)
    # for context in read_lexsub_xml(sys.argv[1]):
    #     # print(context)  # useful for debugging
    #     prediction = word2vec.predict_nearest(context) 
    #     print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))

    ## part5
    bert = BertPredictor()
    for context in read_lexsub_xml(sys.argv[1]):
        # print(context)  # useful for debugging
        prediction = bert.predict(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))

    ## part6 - version1
    # bw = BertWord2VecSubst(W2VMODEL_FILENAME)
    # for context in read_lexsub_xml(sys.argv[1]):
    #     # print(context)  # useful for debugging
    #     prediction = bv.predict(context)
    #     print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))

    ## part6 - version2
    # fw = LeskWord2VecSubst(W2VMODEL_FILENAME)
    # for context in read_lexsub_xml(sys.argv[1]):
    #     # print(context)  # useful for debugging
    #     prediction = fw.predict(context)
    #     print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
    
    """
    part6 - version1 has the best performance but the running time is too long
    """
