# coding: utf-8
import time
import pickle
import javalang
from tokenizers import Tokenizer


class TrieNode():
    def __init__(self):
        self.children = {}
        self.last = False

class Trie():
    def __init__(self, vocab):
        self.root = TrieNode()
        self.vocab = vocab
        self.formTrie((self.vocab.keys()))

    def formTrie(self, keys):
        for key in keys:
            self.insert(key) 
    
    def insert(self, key):
        node = self.root
        for a in key:
            if not node.children.get(a):
                node.children[a] = TrieNode()
            node = node.children[a]
        node.last = True

    def suggestionsRec(self, node, word, results):
        if node.last:
            results.append((word, self.vocab[word]))
        for a, n in node.children.items():
            self.suggestionsRec(n, word + a, results)

    def printAutoSuggestions(self, key, results):
        node = self.root 
        for a in key:
            if not node.children.get(a):
                return 0
            node = node.children[a]

        if not node.children:
            return -1
        
        self.suggestionsRec(node, key, results)
        return 1
