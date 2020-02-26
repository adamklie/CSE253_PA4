import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from torchvision import transforms 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image
from TheRealNapster import *
from data_loader import get_loader 
from tensorboardX import SummaryWriter
from torch.nn.utils.rnn import pack_padded_sequence

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image


def main(args):
    # Image preprocessing
    
    summary = SummaryWriter('{0}/{1}/{2}'.format(args.tensorboard_path, "aya", "uno"))
    
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    encoder = Enigma(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = Christopher(len(vocab), args.embed_size, args.hidden_size, args.num_layers)
    #decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    # Prepare an image
    image = load_image(args.image, transform)
    image_tensor = image.to(device)
    
    # Generate an caption from the image
    feature = encoder(image_tensor)
    #Deterministic
    print("Deterministic Sampling...")
    sampled_ids = decoder.ItsGameTime(feature)
    
    val_loader = val_dataloader = get_loader(args.image_dir,'data/annotations/captions_train2014.json', vocab, transform, 10, shuffle=True, num_workers=2)

    
    #Stochastic:
    #print("Stochastic sampling...")
    #sampled_ids = decoder.IsTheAnswerASmallBoysSundayTrousers(feature)
    #print("finishing...")
    #print(sampled_ids.size())
    #print(sampled_ids)
    sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
    #print(sampled_ids)
    #print(sampled_ids.shape)
    # Convert word_ids to words
    
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)
    
    # Print out the image and the generated caption
    print (sentence)
    image = Image.open(args.image)
    plt.imshow(np.asarray(image))
    
    for i, (image_batch, caption_batch, length_batch) in enumerate(val_loader):
        image_batch = image_batch.to(device)
        caption_batch = caption_batch.to(device)
        target_batch = pack_padded_sequence(caption_batch, length_batch, batch_first=True)[0]
       
        # Run through model
        encoder_features = encoder(image_batch)
        print(encoder_features.size())
        output = decoder(encoder_features, caption_batch, length_batch)

        # Calculate and store loss
        #loss = crit(output, target_batch)
        
        # For the first batch, display how we are doing on four images
        if i == 0:
            un_batch = unnormalize_batch(image_batch[0:3])
            summary.add_images('Epoch [{}/{}] Validation Batch Sample'.format("no_epoch", "sureWhyNot"), un_batch, 0)
            summary.flush()
            caption_ids = decoder.ItsGameTime(encoder_features)
            #print(caption_ids)
            #print(len(caption_ids))
            caption_ids = caption_ids[0:3].cpu().numpy() 
            #caption = 
            caption = get_words(caption_ids, vocab)
            summary.add_text('Epoch [{}/{}] Image 1 Predicted Caption'.format("no_epoch", "sureWhyNot"), caption[0], 1)
            summary.add_text('Epoch [{}/{}] Image 1 Predicted Caption'.format("no_epoch", "sureWhyNot"), caption[1], 1)
            summary.add_text('Epoch [{}/{}] Image 1 Predicted Caption'.format("no_epoch", "sureWhyNot"), caption[2], 1)
            summary.flush()
            break
            
def get_words(captions, vocab):
    sampled_caption = []
    for cap in captions:
        resampleThis = []
        for word_id in cap:
            word = vocab.idx2word[word_id]
            resampleThis.append(word)
            if word == '<end>':
                break
        sampled_caption.append(' '.join(resampleThis))
    for sample in sampled_caption:
        print(sample)
    return sampled_caption
            
    

def unnormalize_batch(image_batch):
    inv_batch = torch.empty(image_batch.size())
    for i, image in enumerate(image_batch):
        inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                                             std=[1/0.229, 1/0.224, 1/0.225])
        orig_image = inv_normalize(image)
        inv_batch[i] = orig_image
    return inv_batch
    
    

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='input image for generating caption')
    parser.add_argument('--image_dir', type=str, default='data/resized2014', help='directory for resized images')
    parser.add_argument('--encoder_path', type=str, default='models/encoder-5-3000.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/decoder-5-3000.ckpt', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--tensorboard_path', type=str, default='tensorboard_comeAtMeBro', help='Directory for Tensorboard output')
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    args = parser.parse_args()
    main(args)
