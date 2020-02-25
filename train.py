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
#from word_holder import word_holder
from build_vocab import Vocabulary

def create_transforms(transform_args):
    train_trans = []
    val_trans = []
    
    for t in transform_args:
        if t == 'crop':
            train_trans.append(transforms.RandomCrop(224))
        elif t == 'hflip':
            train_trans.append(transforms.RandomHorizontalFlip)  
    
    train_trans.append(transforms.ToTensor())
    train_trans.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    
    val_trans.append(transforms.ToTensor())
    val_trans.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(train_trans), transforms.Compose(val_trans)


def val_split(ids, val_fraction, shuffle_dataset=True, random_seed=13):
    dataset_size = len(ids)
    split = int(np.floor(val_fraction * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(ids)
    return ids[split:], ids[:split]


def unnormalize_batch(image_batch):
    inv_batch = torch.empty(image_batch.size())
    for i, image in enumerate(image_batch):
        inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                                             std=[1/0.229, 1/0.224, 1/0.225])
        orig_image = inv_normalize(image)
        inv_batch[i] = orig_image
    return inv_batch
   
    
def val(epoch, data_loader, encode, decode, loss, summary):
    for i, (image_batch, caption_batch, length_batch) in enumerate(data_loader):
        print('Here')
        image_batch.to(device)
        caption_batch.to(device)
        target_batch = pack_padded_sequence(caption_batch, length_batch, batch_first=True)[0]
       
        # Run through model
        encoder_features = encode(image_batch)
        output = decode(encoder_features, caption_batch, length_batch)

        # Calculate and store loss
        loss = loss(output, target_batch)
        
        print('[{}/{}] Epochs, Loss: {:.4f}, Perplexity: {:5.4f}'.\
                      format(epoch, args.num_epochs, loss.item(), np.exp(loss.item())))
        
        # For the first batch, display how we are doing on four images
        if i == 0:
            un_batch = unnormalize_batch(image_batch[0:4])
            summary.add_images('Validation Batch Sample', un_batch, epoch)
            ### add the captions predicted here for this batch
        
        if i == 1:
            break
            
    ###writer.add_scalar('BLEU/Validation, score, epoch)
    summary.add_scalar('Loss/Validation', loss.item(), epoch) 
    summary.flush()

        
def main(args, run_id):
    
    # Create SummaryWriter object for tracking using tensorboard
    writer = SummaryWriter('{0}/{1}'.format(args.tensorboard_path, run_id))
                                                                                                        
    # Load vocab built from build_vocab.py
    with open('data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
        
    # Load IDs that will be used for training
    with open('data/filtered_ids.pkl', 'rb') as f:
        IDs = pickle.load(f)
    
    # Set-up the transforms to apply to the datasets
    train_transforms, val_transforms = create_transforms(args.transforms.split(','))
    
    # Split into training and validation sets
    train_ids, val_ids = val_split(IDs, args.validation_split)
    
    # Load datasets into dataloader
    train_dataloader = get_loader(root=args.image_path, json=args.annotation_path, 
                                  ids=train_ids, vocab=vocab, transform=train_transforms,
                                  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = get_loader(root=args.image_path, json=args.annotation_path, 
                                  ids=val_ids, vocab=vocab, transform=val_transforms,
                                  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    
    # Build models
    encoder = Enigma(args.embed_dim)
    decoder = Christopher(len(vocab), args.embed_dim, 
                          args.units_per_layer, args.num_layers, 
                          args.unit_type, args.pretrained_embedding)
    
    # Add loss function and gradient optimizer
    criterion = nn.CrossEntropyLoss()
    parameters = list(encoder.dudeWheresMyCar.parameters()) + list(encoder.bn1.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    
    # Baseline validation
    #val(0, val_dataloader, encoder, decoder, criterion, writer)
    
    # Train loop
    num_batches = len(train_dataloader)
    for epoch in range(1, args.num_epochs+1):
        for i, (image_batch, caption_batch, length_batch) in enumerate(train_dataloader):
            image_batch.to(device)
            caption_batch.to(device)
            target_batch = pack_padded_sequence(caption_batch, length_batch, batch_first=True)[0]
            
            # Run through model
            encoder_features = encoder(image_batch)
            output = decoder(encoder_features, caption_batch, length_batch)
            
            # Calculate and store loss
            loss = criterion(output, target_batch)
            writer.add_scalar('Loss/Train', loss.item(), epoch) 
            writer.flush()
            
            # Backpropagate
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()
                
            # Print log info
            if i % args.log_step == 0:
                print('[{}/{}] Epochs, [{}/{}] Batches, Loss: {:.4f}, Perplexity: {:5.4f}'.\
                      format(epoch, args.num_epochs, i, num_batches, loss.item(), np.exp(loss.item())))
                
            # Save the model at checkpoints
            if i % args.save_step == 0:
                torch.save(encoder.state_dict(), 
                           os.path.join(args.model_path, 
                                   'Enigma_{0}-{1}_lr-{2}_nl-{3}_hls-{4}_es-{5}_bs-{6}_t-{7}_pte-{8}.ckpt'\
                                   .format(epoch, i, args.learning_rate, args.num_layers, args.units_per_layer, 
                                           args.embed_dim, args.batch_size, args.unit_type, args.pretrained_embedding)
                                  )
                          )
                torch.save(decoder.state_dict(), 
                           os.path.join(args.model_path, 
                                   'Christopher_{0}-{1}_lr-{2}_nl-{3}_hls-{4}_es-{5}_bs-{6}_t-{7}_pte-{8}.ckpt'\
                                   .format(epoch, i, args.learning_rate, args.num_layers, args.units_per_layer, 
                                           args.embed_dim, args.batch_size, args.unit_type, args.pretrained_embedding)
                                  )
                          )
            '''
            if i == 5:
                break
            '''
        ### Probably want a BLEU/BLUE score added in here as well
        #val(epoch, val_dataloader, encoder, decoder, criterion, writer)
    
    writer.close()
       
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # File structure arguments
    path_args = parser.add_argument_group('Input/output options:')
    path_args.add_argument('--tensorboard_path', type=str, default='tensorboard_aklie', help='Directory for Tensorboard output')
    path_args.add_argument('--model_path', type=str, default='models', help='Directory for saved model checkpoints')
    path_args.add_argument('--image_path', type=str, default='data/images/resized', help='Directory with training images')
    path_args.add_argument('--annotation_path', type=str, default='data/annotations/captions_train2014.json', help='Directory with training annotations')
    
    # Data preprocessing arguments
    preproc_args = parser.add_argument_group('Data preprocessing options:')
    preproc_args.add_argument('--transforms', type=str, default='crop', help='Comma separated list of transforms to include in image preprocessing (crop, hflip are supported)')
    
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
    training_args.add_argument('--num_workers', type=int, default=4, help='Number of workers for dataloading')
    training_args.add_argument('--validation_split', type=float, default=0.2, help='Validation split percentage for training')
    training_args.add_argument('--num_epochs', type=int, default=5, help='Number of epochs to train on')
    training_args.add_argument('--learning_rate', type=float, default=5e-3, help='Set learning rate for training')
    
    # Logging arguments
    log_args = parser.add_argument_group('Logging options:')
    log_args.add_argument('--log_step', type=int, default=500, help='Number of batches between printing status')
    log_args.add_argument('--save_step', type=int, default=500, help='Number of batches between saving models')
                           
    args = parser.parse_args()
    
    run_id = 'lr-{0}_nl-{1}_hls-{2}_es-{3}_bs-{4}_t-{5}_pte-{6}'.format(args.learning_rate, args.num_layers,
                                                                        args.units_per_layer, args.embed_dim,
                                                                        args.batch_size, args.unit_type,
                                                                        args.pretrained_embedding)
                           
    if not os.path.exists('{0}/{1}'.format(args.tensorboard_path, run_id)):
        os.makedirs('{0}/{1}'.format(args.tensorboard_path, run_id))
        
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(args, run_id)
