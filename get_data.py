import csv
import os
import argparse
import subprocess
from shutil import copyfile
from pycocotools.coco import COCO
from tqdm import tqdm

def subprocess_cmd(command):
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    
def main(args):
    # Set-up directories
    if not os.path.exists('{}/images'.format(args.data_directory)):
        os.makedirs('{}/images/train'.format(args.data_directory))
        os.makedirs('{}/images/test'.format(args.data_directory))
    
    print('Retrieving annotations')
    if not os.path.exists('{}/annotations'.format(args.data_directory)):
        subprocess_cmd('wget http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip -P {}/'.format(args.data_directory))
        subprocess_cmd('unzip {0}/captions_train-val2014.zip -d {0}/'.format(args.data_directory))
        subprocess_cmd('rm {}/captions_train-val2014.zip'.format(args.data_directory))
          
    # Train images
    coco = COCO('{}/annotations/captions_train2014.json'.format(args.data_directory))
    
    # Read training image ids
    with open(args.train_file, 'r') as f:
        reader = csv.reader(f)
        trainIds = list(reader)
    trainIds = [int(i) for i in trainIds[0]]
    
    # Get images from server using ids
    print('Retrieving training images')
    for img_id in tqdm(trainIds):
        path = coco.loadImgs(img_id)[0]['file_name']
        copyfile('/datasets/COCO-2015/train2014/'+ path, '{0}/images/train/{1}'.format(args.data_directory, path))
        
    # Test images    
    cocoTest = COCO('{}/annotations/captions_val2014.json'.format(args.data_directory))
    
    # Read test image ids
    with open(args.test_file, 'r') as f:
        reader = csv.reader(f)
        testIds = list(reader)  
    testIds = [int(i) for i in testIds[0]]
                 
    # Get images from server using ids
    print('Retrieving test images')
    for img_id in tqdm(testIds):
        path = cocoTest.loadImgs(img_id)[0]['file_name']
        copyfile('/datasets/COCO-2015/val2014/' + path, '{0}/images/test/{1}'.format(args.data_directory, path))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default='TrainImageIds.csv', help='path to file containing training set image IDs')
    parser.add_argument('--test_file', type=str, default='TestImageIds.csv', help='path to file containing test set image IDs')
    parser.add_argument('--data_directory', type=str, default='data', help='path to directory to store data in')
    args = parser.parse_args()
    print('Loading images from {0}, {1} and storing in {2}'.format(args.train_file, args.test_file, args.data_directory))
    main(args)