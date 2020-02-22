import argparse
import os
import torch
from tensorboardX import SummaryWriter


def main(args):
    writer = SummaryWriter('{0}/{1}'.format(args.tensorboard_path, args.run_id))
    # Load vocab and data
    # Build models
    # Add loss function and gradient optimizer
    # Train loop
    for i in range(10):
        writer.add_scalar('Graph', i**2, i)
        writer.flush()
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
       
    args = parser.parse_args()
    
    if not os.path.exists('{0}/{1}'.format(args.tensorboard_path, args.run_id)):
        os.makedirs('{0}/{1}'.format(args.tensorboard_path, args.run_id))
        
    if not os.path.exists('{0}/{1}'.format(args.tensorboard_path, args.run_id)):
        os.makedirs('{0}/{1}'.format(args.tensorboard_path, args.run_id))
        
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(args)