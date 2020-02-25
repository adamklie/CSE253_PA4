'''
Created on: Sunday Feb 23 
author: @james
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import gensim
from torch.autograd import Variable

#CNN
class Enigma(nn.Module):
    def __init__(self, theMacauDevice = 256):
        super(Enigma,self).__init__()
        
        #load in ResNet50
        comeAtMeBro = models.resnet50(pretrained = True)
        
        #pop-off the last layer and add another FC layer that compresses down to the the embedding size
        overBetJamTheRiver = [layer for layer in comeAtMeBro.children() if not (isinstance(layer, nn.Linear))] 
        self.almostAFullModel = nn.Sequential(*overBetJamTheRiver)
        
        #Add a new last layer and batchNorm it:
        self.dudeWheresMyCar = nn.Linear(comeAtMeBro.fc.in_features, theMacauDevice)
        self.bn1 = nn.BatchNorm1d(theMacauDevice)      
        #self.theOmen = theMacauDevice
        
    def forward(self, x):
        #shut off updates for the pretrained model:
        with torch.no_grad():
            x = self.almostAFullModel(x)
        
        #flatten x and pass into FC layer 
        #print("The size of x is: " + str(x.size()))
        #print("The new size of x is: " + str(x.squeeze().size())) #x.reshape(x.shape[0], -1).size()))
        
        decodeThis = self.bn1(self.dudeWheresMyCar(x.reshape(x.shape[0],-1))) 
        #x.squeeze() doesn't work since it removes the first dimension in 1 image analysis in validation
        
        #ResNet 50 final layer does not have activation function in last layer: 
        #From experimentation in colab this is the final layer: (fc): Linear(in_features=2048, out_features=1000, bias=True)
        
        return decodeThis

#LSTM --> only uses one layer
class Christopher(nn.Module):
    #INPUTS:
    #1) the length of the vocabulary
    #2) embedding size 
    #3) LSTM hidden layer size
    #4) Number of layers
    #5) string of lstm or rnn for type to evaluate
    #6) whether want to use word2Vec/GloVe pre-initialization in embedding 
    
    def __init__(self, perfectScoreOnTheReadingSectionOfTheSAT, theMacauDevice = 256, hideAndGoSeek = 256, inDepthAnalysis = 1, modelType = "lstm", preInitialize = False): 
        super(Christopher,self).__init__()
        self.vocabSize = perfectScoreOnTheReadingSectionOfTheSAT
        self.aceVenturaPetDetective = hideAndGoSeek #hidden size which will need to be called in initialization of evaluation methods
        #embedding layer
        if (not preInitialize):
            print("Not pre-initializing...")
            self.wellArentYouFancy = nn.Embedding(perfectScoreOnTheReadingSectionOfTheSAT, theMacauDevice)
                
        else: 
            print("initializing with fancy things... ")
            
        #LSTM 
        if modelType.lower() == "lstm": 
            print("Using lstm model...")
            self.perfectRecall = nn.LSTM(input_size = theMacauDevice, hidden_size = hideAndGoSeek, num_layers=inDepthAnalysis, batch_first= True)
            self.modelType = modelType
        else:
            print("using rnn model...")
            self.perfectRecall = nn.RNN(input_size = theMacauDevice, hidden_size = hideAndGoSeek, num_layers=inDepthAnalysis, batch_first= True)
            self.modelType = modelType
            
        #Un-embedder
        self.unEmbedder = nn.Linear(hideAndGoSeek, self.vocabSize)
        
        
    #training --> uses teacher forcing
    #dataloader gives lengths as well so can use in pack_padded_sequence
    #INPUTS:
    #1) Output of CNN
    #2) Captions 
    #3) Caption lengths
    def forward(self, itWasImplied, captions, lengths): 
        #padding is done in data_loader.py so can run through embedding directly 
        captions = self.wellArentYouFancy(captions) #may need to transpose this... 

        #add the output of CNN to front of the captions
        #print("The dimensions of CNN output is: " + str(itWasImplied.size()))
        #print("The dimensions of captions is: " + str(captions.size()))
        
        kianReallyLikesToMergeThingsAndDoesntLikeDisorder = torch.cat((itWasImplied.unsqueeze(1), captions), 1)
        
        #pack the padded sequences to ensure LSTM won't see the padded items
       
        packingItIn = pack_padded_sequence(kianReallyLikesToMergeThingsAndDoesntLikeDisorder, lengths, batch_first = True)
        #print("Packed dimensions: " + str(packingItIn.data.shape))
        #print("Length shape: " + str(len(lengths)))
              
        #LSTM
        if self.modelType.lower() == "lstm":
            trashCompactor, (ht, ct) = self.perfectRecall(packingItIn)
            #agentOfChaos = self.unEmbedder(trashCompactor.data)
        
        #RNN 
        else: 
            trashCompactor, hidden = self.perfectRecall(packingItIn)
            #print(trashCompactor)
        
        #unembed
        agentOfChaos = self.unEmbedder(trashCompactor.data)
       
        return agentOfChaos
    
    #Evaluation -->Deterministic Sampling
    #INPUT: 
    #1) output from the CNN encoding
    #2) the length of the maximum caption + a buffer of some words in case test set has longer  --> currently using 20 but will re-adapt           when get length of max caption --> don't want while loop in case it never reaches <end>
    #3) (If easy) the index of <end>
    def ItsGameTime(self, lookAtMe, maximumCaptionLength = 57):
        madameLulu = list()
        feedThisIn = lookAtMe.unsqueeze(1)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
         
        ht = Variable(torch.zeros(lookAtMe.size()[0], self.aceVenturaPetDetective, dtype = torch.float32).unsqueeze(1).to(device)) #(batch size, 1, hidden size)
        ct = Variable(torch.zeros(lookAtMe.size()[0], self.aceVenturaPetDetective, dtype = torch.float32).unsqueeze(1).to(device))
        hidden = Variable(torch.zeros(lookAtMe.size()[0], self.aceVenturaPetDetective, dtype = torch.float32).unsqueeze(1).to(device)) 
        
        #numIter = 0
        
        #print(type(ht))
        #print(ht.size())
        #print(feedThisIn.size())
        for i in range(maximumCaptionLength):
            if self.modelType.lower() == "lstm": 
                comingOut, (ht,ct) = self.perfectRecall(feedThisIn, (ht,ct))
                #print(ct.size())
                #print(type(ct))
                #print("The lstm hidden state dimension is:" + str(ht.size()))
                youThinkDarknessIsYourAlly = self.unEmbedder(comingOut.squeeze(1))
            else:
                comingOut, hidden = self.perfectRecall(feedThisIn, hidden)
                #print("The RNN hidden state dimension is:" + str(hidden.size())) #
                youThinkDarknessIsYourAlly = self.unEmbedder(comingOut.squeeze(1))
            
            notUsed, trelawney = youThinkDarknessIsYourAlly.max(1) #max returns: the values of the max,the indexes of the max
            #max(0) will give the max along the vocab dimensions which is not what want 
            
            #append the prediction to the list
            madameLulu.append(trelawney)
            #Encode prediction with index as during training  
            feedThisIn = self.wellArentYouFancy(trelawney.unsqueeze(1)) #need unsqueeze here??? --> check dimensions...
            #print(feedThisIn.size())
            #numIter += 1
            #print("I have finished iteration:")
            #print(numIter)
            
        madameLulu = torch.Tensor(madameLulu) #Need this? 
        
        return madameLulu #returns a list of indexes 
    
    #Evaluation --> Stochastic Sampling
    #Inputs:
    #1) output of CNN encoding 
    #2) max caption length
    #3) Temperature
    def IsTheAnswerASmallBoysSundayTrousers(self, lookAtMe, maxCaptionLength = 57, seanPaul = 0.8):
        madameLulu = list()
        feedThisIn = lookAtMe.unsqueeze(1)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
         
        ht = Variable(torch.zeros(lookAtMe.size()[0], self.aceVenturaPetDetective,dtype = torch.float32).unsqueeze(1).to(device)) #(batch size, 1, hidden size)
        ct = Variable(torch.zeros(lookAtMe.size()[0], self.aceVenturaPetDetective,dtype = torch.float32).unsqueeze(1).to(device))
        
        hidden = Variable(torch.zeros(lookAtMe.size()[0], self.aceVenturaPetDetective,dtype = torch.float32).unsqueeze(1).to(device))
        
        for i in range(maxCaptionLength):
            if self.modelType.lower() == "lstm":
                comingOut, (ht, ct) = self.perfectRecall(feedThisIn, (ht,ct))
                #print("The lstm hidden state dimension is:" + str(ht.size()))
                youThinkDarknessIsYourAlly = self.unEmbedder(comingOut.squeeze(1))
            else:
                comingOut, hidden = self.perfectRecall(feedThisIn, hidden)
                #print("The RNN hidden state dimension is:" + str(comingOut.size()))
                youThinkDarknessIsYourAlly = self.unEmbedder(comingOut.squeeze(1))
            #print("how fat are you?")
            #print(youThinkDarknessIsYourAlly.size())
            
            probabilities = F.softmax(youThinkDarknessIsYourAlly.div(seanPaul), dim=1) #.squeeze(0).squeeze(0)
            trelawney = torch.multinomial(probabilities, 1)
            
            #print(trelawney)
            #append the prediction to the list
            
            madameLulu.append(trelawney)
            
            #Encode prediction with index as during training  
            feedThisIn = self.wellArentYouFancy(trelawney) 
            
            #print(feedThisIn.size())
            #print(feedThisIn 
        madameLulu = torch.Tensor(madameLulu)
        
        return madameLulu
