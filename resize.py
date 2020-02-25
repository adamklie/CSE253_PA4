import argparse
import os
from PIL import Image

def resize_images(image_dir, output_dir, size):
    """Resize the images in 'image_dir' and save into 'output_dir'."""
    
    #create output dir if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    images=os.listdir(image_dir)
    
    for i,image in enumerate(images):
        with Image.open(image_dir+image) as img:
            img=img.resize(size,Image.ANTIALIAS)
            img.save(output_dir+image)
        if (i+1) % 100 ==0:
            print ("{}/{} images resized and saved into {}".format(i+1,len(images), output_dir))
    
    
def main(args):
    image_dir = args.image_dir
    output_dir = args.output_dir
    image_size = [args.image_size, args.image_size]
    resize_images(image_dir, output_dir, image_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='./data/images/train/',
                        help='directory for train images')
    parser.add_argument('--output_dir', type=str, default='./data/images/resized/',
                        help='directory for saving resized images')
    parser.add_argument('--image_size', type=int, default=256,
                        help='size for image after processing')
    args = parser.parse_args()
    main(args)