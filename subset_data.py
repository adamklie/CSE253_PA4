import pickle
import os
import argparse
from pycocotools.coco import COCO
            
def main(args):
    coco = COCO(args.train_annotation_path)
    train_IDs = list(coco.anns.keys())
    train_filtered_ids = list()    
    for i, ann_id in enumerate(train_IDs):
        img_id = coco.anns[ann_id]['image_id']
        image_name = coco.loadImgs(img_id)[0]['file_name']
        
        if os.path.exists(os.path.join(args.train_image_path, image_name)):
            train_filtered_ids.append(ann_id)
            
        if (i+1) % 1000 == 0:
            print('[{}/{}] Train filtered IDs'.format(i+1, len(train_IDs)))
            
        if len(train_filtered_ids) == args.max_train_examples:
            break
            
    with open(args.train_id_path, 'wb') as f:
        pickle.dump(train_filtered_ids, f)
        
    coco = COCO(args.test_annotation_path)
    test_IDs = list(coco.anns.keys())
    test_filtered_ids = list()
    for i, ann_id in enumerate(test_IDs):
        img_id = coco.anns[ann_id]['image_id']
        image_name = coco.loadImgs(img_id)[0]['file_name']
            
        if os.path.exists(os.path.join(args.test_image_path, image_name)):
            test_filtered_ids.append(ann_id)
            
        if (i+1) % 1000 == 0:
            print('[{}/{}] Test filtered IDs'.format(i+1, len(test_IDs)))
            
        if len(test_filtered_ids) == args.max_test_examples:
            break
                             
    with open(args.test_id_path, 'wb') as f:
        pickle.dump(test_filtered_ids, f)
                  
    print("Total IDs saved: Train {}, Test {}".format(len(train_filtered_ids), len(test_filtered_ids)))
    print("Saved the filtered IDs to {}, {}".format(args.train_id_path, args.test_id_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_image_path', type=str, default='./data/images/train', 
                        help='path for directory where train images are kept')
    parser.add_argument('--test_image_path', type=str, default='./data/images/test', 
                        help='path for directory where test images are kept')
    parser.add_argument('--train_annotation_path', type=str, 
                        default='data/annotations/captions_train2014.json', 
                        help='path for train annotation file')
    parser.add_argument('--test_annotation_path', type=str, 
                        default='data/annotations/captions_val2014.json', 
                        help='path for test annotation file')
    parser.add_argument('--train_id_path', type=str, default='./data/train_filtered_ids.pkl', 
                        help='path for saving filtered train ID list')
    parser.add_argument('--test_id_path', type=str, default='./data/test_filtered_ids.pkl', 
                        help='path for saving filtered test ID list')
    parser.add_argument('--max_train_examples', type=int, default=100000, 
                        help='maximum number of training examples to include')
    parser.add_argument('--max_test_examples', type=int, default=100000, 
                        help='maximum number of test examples to include')
    args = parser.parse_args()
    main(args)