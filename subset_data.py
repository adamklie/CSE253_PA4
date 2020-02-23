import pickle
import os
import argparse
from pycocotools.coco import COCO
            
def main(args):
    coco = COCO(args.annotation_path)
    IDs = list(coco.anns.keys())
    filtered_ids = list()
    for i, ann_id in enumerate(IDs):
        img_id = coco.anns[ann_id]['image_id']
        image_name = coco.loadImgs(img_id)[0]['file_name']
        if os.path.exists(os.path.join(args.image_path, image_name)):
            filtered_ids.append(ann_id)
        if (i+1) % 1000 == 0:
            print('[{}/{}] Filtered IDs'.format(i+1, len(IDs)))
        if len(filtered_ids) == args.max_examples:
            break
                  
    with open(args.id_path, 'wb') as f:
        pickle.dump(filtered_ids, f)
                  
    print("Total IDs saved: {}".format(len(filtered_ids)))
    print("Saved the filted IDs to '{}'".format(args.id_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='./data/images/train', 
                        help='path for directory where images are kept')
    parser.add_argument('--annotation_path', type=str, 
                        default='data/annotations/captions_train2014.json', 
                        help='path for train annotation file')
    parser.add_argument('--id_path', type=str, default='./data/filtered_ids.pkl', 
                        help='path for saving filtered ID list')
    parser.add_argument('--max_examples', type=int, default=100000, 
                        help='maximum number of training examples to include')
    args = parser.parse_args()
    main(args)