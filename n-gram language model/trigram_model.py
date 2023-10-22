import sys
from collections import defaultdict
from collections import Counter
import math
import random
import os
import os.path
import numpy as np

"""
COMS W4705 - Natural Language Processing - Fall 2022 
Prorgramming Homework 1 - Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sentence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1 
    """
    # insert START and STOP
    sequence = sentence[:] # do not modify on the original list
    for i in range(max(n-1, 1)):
        sequence.insert(0, 'START')
    sequence.append('STOP')

    # get tuple
    res = []
    for i in range(len(sequence) - n + 1):
        res.append(tuple(sequence[i:i+n]))

    return res


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = Counter()
        self.bigramcounts = Counter()
        self.trigramcounts = Counter()
        self.sentences = 0

        for sentence in corpus:
            self.unigramcounts.update(get_ngrams(sentence, 1))
            self.bigramcounts.update(get_ngrams(sentence, 2))
            self.trigramcounts.update(get_ngrams(sentence, 3))
            self.sentences += 1

        self.unigramall = sum(self.unigramcounts.values()) - self.unigramcounts[('START')] # include stop but not start

        return

    def raw_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        if trigram[0:2]==('START', 'START'): # number appears at the beginning / number of sentences
            return float(self.trigramcounts[trigram]) / self.sentences
        elif trigram[0:2] not in self.bigramcounts.keys(): # unigram
            return float(self.unigramcounts[trigram[0:1]]) / self.unigramall
        else:
            return float(self.trigramcounts[trigram]) / self.bigramcounts[trigram[0:2]]

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        return float(self.bigramcounts[bigram]) / self.unigramcounts[bigram[0:1]]
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  
        return float(self.unigramcounts[unigram]) / self.unigramall


    def generate_sentence(self, t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        context = ('START', 'START')
        sentence = []
        count = 0
        while(1):
            words = []
            probs = []
            for word in self.lexicon:
                if (context + (word,)) in self.trigramcounts.keys():
                    words.append(word)
                    probs.append(self.raw_trigram_probability(context + (word,)))
            probs = [float(i)/sum(probs) for i in probs]
            next = np.random.choice(words, p=probs)
            if next == 'UNK':
                continue
            sentence.append(next)
            count += 1
            context = (context[-1], next)
            if next == 'STOP':
                break
            if count == t:
                break

        return sentence            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        res = lambda1*self.raw_trigram_probability(trigram) + \
            lambda2*self.raw_bigram_probability(trigram[0:2]) + lambda3*self.raw_unigram_probability(trigram[0:1])
        return res
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        res = 0
        for trigram in get_ngrams(sentence, 3):
            res += math.log2(self.smoothed_trigram_probability(trigram))
        return res

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        res = 0
        all_tokens = 0
        for sentence in corpus:
            res += self.sentence_logprob(sentence)
            all_tokens += len(sentence) + 1
        l = float(res) / all_tokens
        return math.pow(2, -1*l)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            if pp1 < pp2:
                correct += 1
            total += 1
    
        for f in os.listdir(testdir2):
            pp1 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            pp2 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            if pp1 < pp2:
                correct += 1
            total += 1
            
        
        return float(correct) / total

if __name__ == "__main__":

    # part1: get ngrams
    seq = ["natural","language","processing"]
    print("\n======== Part1: get n-grams ========")
    print("the original sequence is ", seq)
    print("1-gram: ", get_ngrams(seq, 1))
    print("2-gram: ", get_ngrams(seq, 2))
    print("3-gram: ", get_ngrams(seq, 3))

    # part2: counting n-grams in a corpus
    model = TrigramModel(sys.argv[1]) 
    print("\n======== Part2: count n-grams ========")
    print("occurrence frequency of ('START','START','the') is ", model.trigramcounts[('START','START','the')])
    print("occurrence frequency of ('START','the')         is ", model.bigramcounts[('START','the')])
    print("occurrence frequency of ('the',)                is ", model.unigramcounts[('the',)])

    # part3: generating text
    print("\n======== Part3: generate text ========")
    print("randomly generate 2 sentences")
    print(model.generate_sentence())
    print(model.generate_sentence())

    # part6: perplexity
    print("\n======== Part6: perplexity ========")
    dev_corpus_train = corpus_reader(sys.argv[1], model.lexicon)
    pp_train = model.perplexity(dev_corpus_train)
    dev_corpus_test = corpus_reader(sys.argv[2], model.lexicon)
    pp_test = model.perplexity(dev_corpus_test)
    print("perplexity on the test  set is ", pp_test)
    print("perplexity on the train set is ", pp_train)

    # part7: using the model for text classification
    acc = essay_scoring_experiment('./hw1_data/ets_toefl_data/train_high.txt', \
                                   './hw1_data/ets_toefl_data/train_low.txt', \
                                   './hw1_data/ets_toefl_data/test_high', \
                                   './hw1_data/ets_toefl_data/test_low')
    print("\n======== Part7: accuracy ========")
    print("accuracy of prediction is ", acc, "\n")

    # run the script from the command line with
    # $ python trigram_model.py ./hw1_data/brown_train.txt ./hw1_data/brown_test.txt

 

