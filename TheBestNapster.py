'''
Created on: Sunday Feb 23 
Author: James Talwar
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import gensim
from torch.autograd import Variable

# CNN Encoder: Input the embedding size (what size the new linear layer eeds to output to be consistent with LSTM/RNN format)
class Enigma(nn.Module):
    
    # Initialize the model, takes in embedding dimensions
    def __init__(self, theMacauDevice = 256):
        super(Enigma,self).__init__()
     
        # Load in a pretrainedResNet50
        comeAtMeBro = models.resnet50(pretrained = True)
        
        # Pop-off the last layer and add another FC layer that compresses down to the the embedding size
        overBetJamTheRiver = [layer for layer in comeAtMeBro.children() if not (isinstance(layer, nn.Linear))] 
        self.almostAFullModel = nn.Sequential(*overBetJamTheRiver)
        
        # Add a new last layer and batchNorm it:
        self.dudeWheresMyCar = nn.Linear(comeAtMeBro.fc.in_features, theMacauDevice)
        self.bn1 = nn.BatchNorm1d(theMacauDevice)      
        
    # Feed the input through the model
    def forward(self, x):
        # Shut off updates for the pretrained model:
        with torch.no_grad():
            x = self.almostAFullModel(x)
        
        # Flatten x and pass into FC layer 
        decodeThis = self.bn1(self.dudeWheresMyCar(x.reshape(x.shape[0],-1))) 
        
        #ResNet 50 final layer does not have activation function in last layer: 
        #From experimentation in colab this is the final layer: (fc): Linear(in_features=2048, out_features=1000, bias=True)x
        return decodeThis

    
# LSTM/RNN Decode --> Sequentially passes embedded words into recurrent units (can be specified as RNN or LSTM units)
class Christopher(nn.Module):
    
    # Initialize the model:
    # wordzzz: the vocabulary
    # theMacauDevice: embedding size 
    # hideAndGoSeek: LSTM hidden layer size
    # inDepthAnalysis: Number of layers
    # modelType: string of lstm or rnn for type to evaluate
    # preInitialize: whether want to use word2Vec/GloVe pre-initialization in embedding 
    
    def __init__(self, wordzzz, theMacauDevice = 256, hideAndGoSeek = 256, inDepthAnalysis = 1, 
                 modelType = "lstm", preInitialize = False): 
        super(Christopher,self).__init__()
        self.vocabSize = len(wordzzz)
        self.aceVenturaPetDetective = hideAndGoSeek
        
        # Embedding layer
        if (not preInitialize):
            print("Not pre-initializing...")
            self.wellArentYouFancy = nn.Embedding(self.vocabSize, theMacauDevice)
                
        else: 
            print("Initializing with fancy things... ")
            #weights matrix
            weights_matrix=np.zeros((self.vocabSize, 256))
            vocab_words=[x for x in wordzzz.word_to_idx.keys()]
            
            words_found=0
            pretrained=gensim.models.KeyedVectors.load_word2vec_format('data/word2vec.model.bin', binary=True)
            for i,word in enumerate(vocab_words):
                try:
                    word_id=wordzzz.word_to_idx[word]
                    weights_matrix[word_id]=pretrained[word]
                    words_found +=1
                except KeyError:
                    word_id=wordzzz.word_to_idx[word]
                    weights_matrix[word_id]=np.random.random_sample(256)
            print("[{}/{}] words found.".format(words_found, len(vocab_words)))
            num_embeddings, embedding_dim = self.vocabSize,256
            weights=torch.FloatTensor(weights_matrix)
            self.wellArentYouFancy = nn.Embedding(num_embeddings, embedding_dim)
            self.wellArentYouFancy.from_pretrained(weights)
                    
        # LSTM vs RNN 
        if modelType.lower() == "lstm": 
            print("Using lstm model...")
            self.perfectRecall = nn.LSTM(input_size=theMacauDevice, hidden_size=hideAndGoSeek, 
                                         num_layers=inDepthAnalysis, batch_first=True)
            self.modelType = modelType
            
        else:
            print("Using rnn model...")
            self.perfectRecall = nn.RNN(input_size=theMacauDevice, hidden_size=hideAndGoSeek, 
                                        num_layers=inDepthAnalysis, batch_first= True)
            self.modelType = modelType
            
        # un-Embedder
        self.unEmbedder = nn.Linear(hideAndGoSeek, self.vocabSize)
        
    # Training --> uses teacher forcing, dataloader gives lengths as well so can use in pack_padded_sequence
    # itWasImplied: Output of CNN
    # captions: Captions 
    # lengths: Caption lengths
    def forward(self, itWasImplied, captions, lengths): 
        
        #padding is done in data_loader.py so can run through embedding directly 
        captions = self.wellArentYouFancy(captions)

        #add the output of CNN to front of the captions
        kianReallyLikesToMergeThingsAndDoesntLikeDisorder = torch.cat((itWasImplied.unsqueeze(1), captions), 1)
        
        #pack the padded sequences to ensure LSTM won't see the padded items     
        packingItIn = pack_padded_sequence(kianReallyLikesToMergeThingsAndDoesntLikeDisorder, lengths, batch_first = True)
              
        # LSTM
        if self.modelType.lower() == "lstm":
            trashCompactor, (ht, ct) = self.perfectRecall(packingItIn)
        
        # RNN 
        else: 
            trashCompactor, hidden = self.perfectRecall(packingItIn)
        
        # un-Embed
        agentOfChaos = self.unEmbedder(trashCompactor.data)
       
        return agentOfChaos
    
    # Evaluation -->Deterministic Sampling
    # lookAtMeoutput from the CNN encoding    
    # numLayers: number of hidden layers passed in from the model
    # maximumCaptionLength: the length of the maximum caption + a buffer of some words in case test set has longer
    def ItsGameTime(self, lookAtMe, numLayers, maximumCaptionLength = 57):
        madameLulu = list()
        feedThisIn = lookAtMe.unsqueeze(1)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
         
        ht = Variable(torch.zeros(numLayers, lookAtMe.size()[0], 
                                  self.aceVenturaPetDetective,dtype = torch.float32).to(device))
        ct = Variable(torch.zeros(numLayers, lookAtMe.size()[0], 
                                  self.aceVenturaPetDetective,dtype = torch.float32).to(device))
        hidden = Variable(torch.zeros(numLayers, lookAtMe.size()[0], 
                                      self.aceVenturaPetDetective,dtype = torch.float32).to(device)) 
        
        for i in range(maximumCaptionLength):
            if self.modelType.lower() == "lstm": 
                comingOut, (ht,ct) = self.perfectRecall(feedThisIn, (ht,ct))
                youThinkDarknessIsYourAlly = self.unEmbedder(comingOut.squeeze(1))
            else:
                comingOut, hidden = self.perfectRecall(feedThisIn, hidden)
                youThinkDarknessIsYourAlly = self.unEmbedder(comingOut.squeeze(1))
            
            notUsed, trelawney = youThinkDarknessIsYourAlly.max(1)
            madameLulu.append(trelawney)
            feedThisIn = self.wellArentYouFancy(trelawney.unsqueeze(1))  ### Needed?
        
        madameLulu = torch.stack(madameLulu, dim = 1)
        
        return madameLulu
    
    # Evaluation --> Stochastic Sampling:
    # lookAtMeoutput from the CNN encoding    
    # numLayers: number of hidden layers passed in from the model
    # maximumCaptionLength: the length of the maximum caption + a buffer of some words in case test set has longer
    # seanPaul: temperature
    def IsTheAnswerASmallBoysSundayTrousers(self, lookAtMe, numLayers, seanPaul = 0.8, maxCaptionLength = 57):
        madameLulu = list()
        feedThisIn = lookAtMe.unsqueeze(1)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
         
        ht = Variable(torch.zeros(numLayers, lookAtMe.size()[0], 
                                  self.aceVenturaPetDetective,dtype = torch.float32).to(device))
        ct = Variable(torch.zeros(numLayers, lookAtMe.size()[0], 
                                  self.aceVenturaPetDetective,dtype = torch.float32).to(device))
        hidden = Variable(torch.zeros(numLayers, lookAtMe.size()[0], 
                                      self.aceVenturaPetDetective,dtype = torch.float32).to(device)) 
        
        for i in range(maxCaptionLength):
            if self.modelType.lower() == "lstm":
                comingOut, (ht, ct) = self.perfectRecall(feedThisIn, (ht,ct))
                youThinkDarknessIsYourAlly = self.unEmbedder(comingOut.squeeze(1))
            else:
                comingOut, hidden = self.perfectRecall(feedThisIn, hidden)
                youThinkDarknessIsYourAlly = self.unEmbedder(comingOut.squeeze(1))
            
            probabilities = F.softmax(youThinkDarknessIsYourAlly.div(seanPaul), dim=1)
            trelawney = torch.multinomial(probabilities, 1)            
            madameLulu.append(trelawney.squeeze(1))  
            feedThisIn = self.wellArentYouFancy(trelawney)  ### Needed?
            
        madameLulu = torch.stack(madameLulu, dim = 1)
        
        return madameLulu