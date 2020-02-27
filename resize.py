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

def max_crop_resize(image_dir, output_dir, size):
    """Crop the maximum square from the images and resize them. Read from'image_dir' and save into 'output_dir'."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    images=os.listdir(image_dir)
    
    for i,image in enumerate(images):
        with Image.open(image_dir+image) as img:
            img = crop_max_square(img)
            img = img.resize(size,Image.LANCZOS)
            img.save(output_dir+image, quality = 95)
        if (i+1) % 100 ==0:
            print ("{}/{} images resized and saved into {}".format(i+1,len(images), output_dir))
#             break

# Copied from https://note.nkmk.me/en/python-pillow-image-crop-trimming/
def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

# Copied from https://note.nkmk.me/en/python-pillow-image-crop-trimming/
def crop_max_square(pil_img):
    return crop_center(pil_img, min(pil_img.size), min(pil_img.size))

def main(args):
    train_image_dir = args.train_image_dir
    train_output_dir = args.train_output_dir
    test_image_dir = args.test_image_dir
    test_output_dir = args.test_output_dir
    image_size = [args.image_size, args.image_size]
    max_crop_resize(train_image_dir, train_output_dir, image_size)
    max_crop_resize(test_image_dir, test_output_dir, image_size)    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
#     parser.add_argument('--image_dir', type=str, default='./data/images/train/',
#                         help='deprecated-directory for train images')
#     parser.add_argument('--output_dir', type=str, default='./data/images/resized/',
#                         help='deprecated-directory for saving resized images')
    parser.add_argument('--image_size', type=int, default=256,
                        help='size for image after processing')
    parser.add_argument('--train_image_dir', type=str, default='./data/images/train/',
                        help='directory for train images')
    parser.add_argument('--train_output_dir', type=str, default='./data/images/train_resized/',
                        help='directory for saving resized and cropped train images')

    parser.add_argument('--test_image_dir', type=str, default='./data/images/test/',
                        help='directory for test images')
    parser.add_argument('--test_output_dir', type=str, default='./data/images/test_resized/',
                        help='directory for saving resized and cropped test images')


    args = parser.parse_args()
    main(args)