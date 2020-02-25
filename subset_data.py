import pickle
import os
import argparse
from pycocotools.coco import COCO
            
def main(args):
    coco = COCO(args.annotation_path)
    IDs = list(coco.anns.keys())
    train_filtered_ids = list()
    test_filtered_ids = list()
    
    in_train = False
    in_test = False
    
    for i, ann_id in enumerate(IDs):
        img_id = coco.anns[ann_id]['image_id']
        image_name = coco.loadImgs(img_id)[0]['file_name']
        
        if os.path.exists(os.path.join(args.train_image_path, image_name)):
            train_filtered_ids.append(ann_id)
            in_train = True
            
        if os.path.exists(os.path.join(args.test_image_path, image_name)):
            test_filtered_ids.append(ann_id)
            in_test = True
            
        if in_train and in_test:
            print('Warning: training set and test set are overlapping')
            print(image_name, 'in both')
        
        in_train = False
        in_test = False
            
        if (i+1) % 1000 == 0:
            print('[{}/{}] Filtered IDs'.format(i+1, len(IDs)))
        if len(train_filtered_ids) == args.max_examples:
            break
                  
    with open(args.train_id_path, 'wb') as f:
        pickle.dump(train_filtered_ids, f)
        
    with open(args.test_id_path, 'wb') as f:
        pickle.dump(test_filtered_ids, f)
                  
    print("Total IDs saved: Train {}, Test {}".format(len(train_filtered_ids), len(test_filtered_ids)))
    print("Saved the filted IDs to '{}'".format(args.train_id_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_image_path', type=str, default='./data/images/train', 
                        help='path for directory where train images are kept')
    parser.add_argument('--test_image_path', type=str, default='./data/images/test', 
                        help='path for directory where test images are kept')
    parser.add_argument('--annotation_path', type=str, 
                        default='data/annotations/captions_train2014.json', 
                        help='path for train annotation file')
    parser.add_argument('--train_id_path', type=str, default='./data/train_filtered_ids.pkl', 
                        help='path for saving filtered train ID list')
    parser.add_argument('--test_id_path', type=str, default='./data/test_filtered_ids.pkl', 
                        help='path for saving filtered test ID list')
    parser.add_argument('--max_examples', type=int, default=100000, 
                        help='maximum number of training examples to include')
    args = parser.parse_args()
    main(args)