import argparse
import os
import numpy as np
import pickle
from word_holder import build_word_holder
from pycocotools.coco import COCO
from data_loader import get_loader
import torch
from tensorboardX import SummaryWriter
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from TheRealNapster import *
from word_holder import word_holder
    

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
            resampleThis.append(word)
            if word == '<end>':
                break
        sampled_caption.append(' '.join(resampleThis))
    for sample in sampled_caption:
        print(sample)
    return sampled_caption
      

def main(args, run_id):
    # Create SummaryWriter object for tracking using tensorboard
    writer = SummaryWriter('{0}/{1}/{2}'.format(args.tensorboard_path, args.run_id, run_id))
    
    test_transforms = transforms.Compose([
        transforms.RandomCrop(200),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocab built from build_vocab.py
    with open('data/vocabulary.pkl', 'rb') as f:
        vocab = pickle.load(f)

    # Load IDs that will be used for training
    with open('data/test_filtered_ids.pkl', 'rb') as f:
        IDs = pickle.load(f)

    print(len(IDs))
    test_dataloader = get_loader(root=args.image_path, json=args.annotation_path, 
                                  ids=IDs, vocab=vocab, transform=test_transforms,
                                  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    
    # Build models
    encoder = Enigma(args.embed_dim).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = Christopher(len(vocab), args.embed_dim, args.units_per_layer, args.num_layers)
    
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    encoder.load_state_dict(torch.load('models/Enigma_5-500_lr-0.005_nl-1_hls-256_es-256_bs-128_t-lstm_pte-False.ckpt'))
    decoder.load_state_dict(torch.load('models/Christopher_5-500_lr-0.005_nl-1_hls-256_es-256_bs-128_t-lstm_pte-False.ckpt'))

    for i, (image_batch, caption_batch, length_batch) in enumerate(test_dataloader):
        print('here')
        image_batch = image_batch.to(device)
        caption_batch = caption_batch.to(device)
        target_batch = pack_padded_sequence(caption_batch, length_batch, batch_first=True)[0]
       
        # Run through model
        encoder_features = encoder(image_batch)
        print(encoder_features.size())
        output = decoder(encoder_features, caption_batch, length_batch)
        
        # For the first batch, display how we are doing on four images
        if i == 0:
            un_batch = unnormalize_batch(image_batch[0:3])
            print(un_batch.size())
            writer.add_images('Epoch [{}/{}] Validation Batch Sample'.format(0, 0), un_batch, 0)
            caption_ids = decoder.ItsGameTime(encoder_features)
            caption_ids = caption_ids[0:3].cpu().numpy() 
            caption = get_words(caption_ids, vocab)
            writer.add_text('Epoch [{}/{}] Image 1 Predicted Caption'.format(0,0), caption[0], 1)
            writer.add_text('Epoch [{}/{}] Image 1 Predicted Caption'.format(0,0), caption[1], 1)
            writer.add_text('Epoch [{}/{}] Image 1 Predicted Caption'.format(0,0), caption[2], 1)
            writer.flush()
            break
    writer.close()
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # File structure arguments
    path_args = parser.add_argument_group('Input/output options:')
    path_args.add_argument('--run_id', type=str, default='test', help='tensorboard subdirectory')
    path_args.add_argument('--tensorboard_path', type=str, default='tensorboard_aklie', help='Directory for Tensorboard output')
    path_args.add_argument('--model_path', type=str, default='models', help='Directory for saved model checkpoints')
    path_args.add_argument('--image_path', type=str, default='data/images/test', help='Directory with test images')
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
    training_args.add_argument('--num_epochs', type=int, default=5, help='Number of epochs to train on')
    training_args.add_argument('--learning_rate', type=float, default=5e-3, help='Set learning rate for training')
    
    # Logging arguments
    log_args = parser.add_argument_group('Logging options:')
    log_args.add_argument('--log_step', type=int, default=10, help='Number of batches between printing status')
    log_args.add_argument('--save_step', type=int, default=500, help='Number of batches between saving models')
                           
    args = parser.parse_args()
    
    run_id = 'lr-{0}_nl-{1}_hls-{2}_es-{3}_bs-{4}_t-{5}_pte-{6}'.format(args.learning_rate, args.num_layers,
                                                                        args.units_per_layer, args.embed_dim,
                                                                        args.batch_size, args.unit_type,
                                                                        args.pretrained_embedding)
                           
    if not os.path.exists('{0}/{1}'.format(args.tensorboard_path, run_id)):
        os.makedirs('{0}/{1}'.format(args.tensorboard_path, run_id))
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    main(args, run_id)
