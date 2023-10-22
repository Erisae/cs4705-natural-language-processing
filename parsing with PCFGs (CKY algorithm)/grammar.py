"""
COMS W4705 - Natural Language Processing - Spring 2023
Homework 2 - Parsing with Context Free Grammars 
Daniel Bauer
"""

import sys
from collections import defaultdict
from math import fsum, isclose

class Pcfg(object): 
    """
    Represent a probabilistic context free grammar. 
    """

    def __init__(self, grammar_file): 
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None 
        self.read_rules(grammar_file)      
 
    def read_rules(self,grammar_file):
        
        for line in grammar_file: 
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line: 
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else: 
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()
                    
     
    def parse_rule(self,rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";",1) 
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)

    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False. 
        """
        # Part 1
        # each rule corresponds to one of the formats permitted in CNF
        nonterminals = self.lhs_to_rules.keys()
        terminals = []
        for rhs in self.rhs_to_rules.keys():
            for item in rhs:
                if item not in nonterminals and item not in terminals:
                    terminals.append(item)

        
        for rules in self.lhs_to_rules.values():
            prob_sum_lhs = 0
            for rule in rules:
                prob_sum_lhs += rule[2]
                if rule[0] not in nonterminals:
                    return False
                if len(rule[1])==2 and not(rule[1][0] in nonterminals and rule[1][1] in nonterminals):
                    return False
                if len(rule[1])==1 and not(rule[1][0] in terminals):
                    return False
                if len(rule[1])==2 and (rule[1][0]==self.startsymbol or rule[1][1]==self.startsymbol):
                    return False

            # all probabilities for the same lhs symbol sum to 1.0, isclose
            if not isclose(prob_sum_lhs, 1):
                print(5, prob_sum_lhs)
                return False

        return True


if __name__ == "__main__":
    with open(sys.argv[1],'r') as grammar_file:
        grammar = Pcfg(grammar_file)

######## part1 reading the grammar and getting started ########
        if not grammar.verify_grammar():
            print("Error: The grammar is not a valid PCFG in CNF.")
        else:
            print("The grammar is a valid PCFG in CNF.")
        
