import argparse
import os
import numpy as np
import pickle
from build_vocab import Vocabulary
from pycocotools.coco import COCO
from data_loader import get_loader
import torch
from tensorboardX import SummaryWriter
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader


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


def unnormalize_batch(image_batch):
    inv_batch = torch.empty(image_batch.size())
    for i, image in enumerate(image_batch):
        inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                                             std=[1/0.229, 1/0.224, 1/0.225])
        orig_image = inv_normalize(image)
        inv_batch[i] = orig_image
    return inv_batch
    
    
def main(args):
    writer = SummaryWriter('{0}/{1}'.format(args.tensorboard_path, args.run_id))
    
    # Load vocab built from build_vocab.py
    with open('data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
        
    # Load IDs that will be used for training
    with open('data/filtered_ids.pkl', 'rb') as f:
        IDs = pickle.load(f)
    
    # Set-up the transforms to apply to the datasets
    train_transforms, val_transforms = create_transforms(args.transforms.split(','))
    
    validation_split = args.validation_split
    shuffle_dataset = True
    random_seed = 13
    dataset_size = len(IDs)
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(IDs)
    train_ids, val_ids = IDs[split:], IDs[:split]
    
    # Load datasets into dataloader
    train_dataloader = get_loader(root=args.image_path, json=args.annotation_path, 
                                  ids=train_ids, vocab=vocab, transform=train_transforms,
                                  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = get_loader(root=args.image_path, json=args.annotation_path, 
                                  ids=val_ids, vocab=vocab, transform=val_transforms,
                                  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    
    # Build models
    
    # Add loss function and gradient optimizer
    
    # Train loop
    for epoch in range(args.num_epochs):
        for i, (image_batch, caption_batch, length_batch) in enumerate(val_dataloader):
            un_batch = unnormalize_batch(image_batch)
            writer.add_images('Batch sample', un_batch, i)
            writer.flush()
            break
    writer.close()
        # Get batch
        # Forward
        # Backward
        # Optimize
        # Print log info
        # Save model
    # Validation step (maybe call function)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=str, default='test', help='Identifier for Tensorboard, default is "test"')
    parser.add_argument('--tensorboard_path', type=str, default='tensorboard_aklie', help='Directory for Tensorboard output')
    parser.add_argument('--model_path', type=str, default='models', help='Directory for saved model checkpoints')
    ### WILL NEED TO CHANGE TO RESIZED
    parser.add_argument('--image_path', type=str, default='data/images/resized', help='Directory with training images')
    parser.add_argument('--annotation_path', type=str, default='data/annotations/captions_train2014.json', help='Directory with training images')
    parser.add_argument('--transforms', type=str, default='crop', help='Comma separated list of transforms to include in image preprocessing (crop, hflip are supported)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for mini-batch gradient descent')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for dataloading')
    parser.add_argument('--validation_split', type=float, default=0.2, help='Validation split percentage for training')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs to train on')
    args = parser.parse_args()
    
    if not os.path.exists('{0}/{1}'.format(args.tensorboard_path, args.run_id)):
        os.makedirs('{0}/{1}'.format(args.tensorboard_path, args.run_id))
        
    if not os.path.exists('{0}/{1}'.format(args.tensorboard_path, args.run_id)):
        os.makedirs('{0}/{1}'.format(args.tensorboard_path, args.run_id))
        
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(args)