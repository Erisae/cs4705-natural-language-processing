from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import keras
import tensorflow as tf

from extract_training_data import FeatureExtractor, State

tf.compat.v1.disable_eager_execution()

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        
        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    '''
    input: a list of words, pos tags in the input sentence
    return: an instance of DependencyStructure
        2. Algorithm
            when buffer not empty, use feature extractor to obtain a representation of the current state
            call model.predict(features) and retrieve a softmax actived vector of possible actions. 
            select the highest scoring permitted transition 
                (create a list of possible actions and sort it according to their output probability, go through and find legal)
            a.arc-left or arc-right are not permitted the stack is empty
            b.Shifting the only word out of the buffer is also illegal, unless the stack is empty
            c.the root node must never be the target of a left-arc
    '''
    def parse_sentence(self, words, pos):
        # initialization done
        state = State(range(1,len(words)))
        state.stack.append(0)    

        while state.buffer: 
            # TODO: Write the body of this loop for part 4 
            features = self.extractor.get_input_representation(words, pos, state)
            output = self.model.predict(features.reshape(1, -1))
            pactions = {key: value for key, value in zip(self.output_labels.values(), output.squeeze().tolist())} # action, possibility
            pactions = dict(sorted(pactions.items(), key=lambda x: x[1], reverse=True)) # sorted
            for action in pactions.keys():
                if (len(state.stack)==0) and (action[0]=='left_arc' or action[0]=='right_arc'):
                    continue
                if len(state.buffer)==1 and action[0]=='shift' and not len(state.stack)==0:
                    continue
                if len(state.stack)>0 and state.stack[-1]==0 and action[0]=='left_arc':
                    continue

                transition = action
                break
            
            if transition[0]=='left_arc':
                state.left_arc(transition[1])
            elif transition[0]=='right_arc':
                state.right_arc(transition[1])
            elif transition[0]=='shift':
                state.shift()

        result = DependencyStructure()
        for p,c,r in state.deps: 
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p,r))
        return result 
        

if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file: 
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
        
