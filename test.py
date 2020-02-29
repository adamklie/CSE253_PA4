'''
Created on: Sunday Feb 23 
Author: Kian Kahlor
'''

import argparse
import os
import numpy as np
import pickle
from word_holder import build_word_holder
from pycocotools.coco import COCO
from data_loader import *
import torch
from tensorboardX import SummaryWriter
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from TheBestNapster import *
from word_holder import word_holder
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction    


def unnormalize_batch(image_batch):
    inv_batch = torch.empty(image_batch.size())
    for i, image in enumerate(image_batch):
        inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                                             std=[1/0.229, 1/0.224, 1/0.225])
        orig_image = inv_normalize(image)
        inv_batch[i] = orig_image
    return inv_batch
  
    
def get_words(captions, vocab):
    sampled_caption = []
    for cap in captions:
        resampleThis = []
        for word_id in cap:
            word = vocab.idx_to_word[word_id]
            if word == '<start>':
                continue
            if word == '<end>':
                break
            resampleThis.append(word)
        sampled_caption.append(" ".join(resampleThis))
    return sampled_caption

      
def main(args, run_id):
    
    # Create SummaryWriter object for tracking using tensorboard
    writer = SummaryWriter('{0}/{1}/{2}'.format(args.tensorboard_path, run_id, args.run_id))

    test_transforms = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

    # Load vocab built from build_vocab.py
    with open('data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    # test image IDs
    test_img_id_path = 'TestImageIds.csv'
    test_cocoloader = CocoTestDataset(root=args.image_path, json=args.annotation_path,
                                   img_id_path=test_img_id_path, vocab=vocab, transform=test_transforms)

    # Build models
    encoder = Enigma(args.embed_dim).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = Christopher(vocab, args.embed_dim, 
                          args.units_per_layer, args.num_layers, 
                          args.unit_type, args.pretrained_embedding).to(device)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    # Initialization for BLEU scores
    score1 = 0
    score4 = 0
    smoother = SmoothingFunction()

    for i in range(len(test_cocoloader)):
        image, caption_list = test_cocoloader[i]
        image_batch = image.unsqueeze(0)
        image_batch = image_batch.to(device)

        # Run through model
        encoder_features = encoder(image_batch)
        
        if args.sampling_type == 'deterministic':
                output = decoder.ItsGameTime(encoder_features, args.num_layers)
        elif args.sampling_type == 'stochastic':
            output = decoder.IsTheAnswerASmallBoysSundayTrousers(encoder_features, args.num_layers, args.temperature)
        else:
            print('What the sampling function do you think you are doing')
            return

        gen_caption = get_words(output.cpu().numpy(), vocab)

        score1 += sentence_bleu(caption_list, gen_caption[0], weights=(1, 0, 0, 0), smoothing_function=smoother.method1)
        score4 += sentence_bleu(caption_list, gen_caption[0], weights=(0, 0, 0, 1), smoothing_function=smoother.method1)

        if i % 100 == 0:
            print(i)
            print(caption_list)
            print(gen_caption)
            print(sentence_bleu(caption_list, gen_caption[0], weights=(1, 0, 0, 0), smoothing_function=smoother.method1))
            print(sentence_bleu(caption_list, gen_caption[0], weights=(0, 0, 0, 1), smoothing_function=smoother.method1))
            
       
    bleu1 = 100*score1/len(test_cocoloader)
    bleu4 = 100*score4/len(test_cocoloader)
    writer.add_text('Average BLEU1 score', str(bleu1), 1)
    writer.add_text('Average BLEU4 score', str(bleu4), 1)
    print('BLEU1 score: ' + str(bleu1))
    print('BLEU4 score: ' + str(bleu4))
    writer.close()
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # File structure arguments
    path_args = parser.add_argument_group('Input/output options:')
    path_args.add_argument('--run_id', type=str, default='test', help='tensorboard subdirectory')
    path_args.add_argument('--tensorboard_path', type=str, default='tensorboard_aklie', help='Directory for Tensorboard output')
    path_args.add_argument('--encoder_path', type=str, 
                           default='models/Enigma_10-500_lr-0.005_nl-1_hls-256_es-256_bs-128_t-lstm_pte-False.ckpt',
                           help='Directory to load saved encoder from')
    path_args.add_argument('--decoder_path', type=str, 
                           default='models/Christopher_10-500_lr-0.005_nl-1_hls-256_es-256_bs-128_t-lstm_pte-False.ckpt',
                           help='Directory to load saved encoder from')
    path_args.add_argument('--image_path', type=str, default='data/images/test_resized', help='Directory with test images')
    path_args.add_argument('--annotation_path', type=str, default='data/annotations/captions_val2014.json', help='Directory with training annotations')
    
    # Model structure arguments
    model_args = parser.add_argument_group('Model structure options:')
    model_args.add_argument('--embed_dim', type=int, default=256, help='Dimensions of word embedding to use')
    model_args.add_argument('--num_layers', type=int, default=1, help='Number of hidden layers in model')
    model_args.add_argument('--units_per_layer', type=int, default=256, help='Number of hidden units in each hidden layer')
    model_args.add_argument('--unit_type', type=str, default='lstm', help='Defines unit, either lstm or rnn')
    model_args.add_argument('--pretrained_embedding', type=bool, default=False, help='Boolean flag for pretrained embeddings')
    
    # Training arguments
    training_args = parser.add_argument_group('Training options:')
    training_args.add_argument('--batch_size', type=int, default=128, help='Batch size for mini-batch gradient descent')
    training_args.add_argument('--num_workers', type=int, default=2, help='Number of workers for dataloading')
    training_args.add_argument('--validation_split', type=float, default=0.2, help='Validation split percentage for training')
    training_args.add_argument('--learning_rate', type=float, default=0.005, help='Set learning rate for training')
    training_args.add_argument('--sampling_type', type=str, default='deterministic', help='Caption generation method (deterministic or stochastic)')
    training_args.add_argument('--temperature', type=float, default=0.8, help='temperature for stochastic sampling, only used if --sampling_type is stochastic')
    
    args = parser.parse_args()
    
    run_id = 'lr-{0}_nl-{1}_hls-{2}_es-{3}_bs-{4}_t-{5}_pte-{6}_st-{7}_temp-{8}'.format(args.learning_rate, args.num_layers,
                                                                        args.units_per_layer, args.embed_dim,
                                                                        args.batch_size, args.unit_type,
                                                                        args.pretrained_embedding, args.sampling_type,
                                                                                        args.temperature)
                           
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    main(args, run_id)