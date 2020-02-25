import pickle
import argparse
import nltk
import pandas as pd

from collections import Counter
from pycocotools.coco import COCO

class word_holder(object):
    
    def __init__(self):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.idx = 0
        self.size = len(self.word_to_idx)
        
    def add_word(self, word):
        
        # if new word during vocab building, create new key
        if word not in self.word_to_idx:
            self.word_to_idx[self.idx] = self.idx
            self.idx_to_word[self.idx] = word
            self.idx += 1
    
    def __call__(self, word):
        
        # check if word is not part of learned vocab, don't throw error
        if word not in self.word_to_idx:
            return self.word_to_idx['<err>']
        else:
            return self.word_to_idx[word]
    
    def __len__(self):
        return len(self.word_to_idx)
    
def build_word_holder(json, rarity):
    
    # read in JSON
    coco = COCO(json)
    # slice captions
    captions = pd.DataFrame.from_dict(coco.anns)
    captions = list(captions.loc['caption'])
    
    # tokenize captions and word count
    vocab_freq = Counter()
    for caption in captions:
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        vocab_freq.update(tokens)
    
    # select words that are not rare
    words = [word for word, freq in vocab_freq.items() if freq >= rarity]

    # build word_holder
    wh = word_holder()
    # pass in all words that met rarity threshold
    for word in words:
        wh.add_word(word)
    
    # add special tokens
    wh.add_word('<pad>')
    wh.add_word('<start>')
    wh.add_word('<end>')
    wh.add_word('<err>')
    
    return wh

if __name__ == '__main__':
    json_dir = 'data/annotations/captions_val2014.json'
    rarity = 10
    wh = build_word_holder(json = json_dir, rarity = 4)
    
    # write binary
    with open('data/vocabulary.pkl', "wb") as f:
        pickle.dump(wh, f)   
            