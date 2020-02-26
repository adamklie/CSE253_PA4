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
            self.word_to_idx[word] = self.idx
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
    
def build_word_holder(json,imageids, rarity):
    
    # read in JSON
    coco = COCO(json)
    # slice captions
    captions = pd.DataFrame.from_dict(coco.anns)
    
    #get filtered captions
    captions=captions.T
    ids=pd.read_csv(imageids)
    ids=[int(x) for x in ids.columns]
    
    filt_captions = captions[captions["image_id"].isin(ids)]["caption"].tolist()
    
    print("[{}/{}] captions remain after filtering.".format(len(filt_captions), len(captions)))
    
    # tokenize captions and word count
    vocab_freq = Counter()
    for i,caption in enumerate(filt_captions):
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        vocab_freq.update(tokens)
        
        if (i+1) % 100 == 0:
            print("[{}/{}] Tokenized the captions.".format(i+1, len(filt_captions)))
    
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

def main(args):
    vocab = build_word_holder(json=args.caption_path,imageids=args.image_path,rarity=args.threshold)
    with open(args.vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(args.vocab_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, 
                        default='./data/annotations/captions_train2014.json', 
                        help='path for train annotation file')
    parser.add_argument('--image_path', type=str, default='TrainImageIds.csv', 
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl', 
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4, 
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)
