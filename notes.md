# Notes

1. Annotations from tutorial are the same as annotations we download (cuz they ripped the code directly). You can confirm with a diff command
2. We have 16556 training images (out of 82785, 1/5th), but over 414113 captions in the json file. While we do have multiple captions per image, not every caption has a corresponding image in the subset that we took. So we need to filter out those captions for images that we will not be using for training. subset_data.py does just this and saves a list of filtered ids that can be used in data loading.
3. The dataLoader class from PyTorch will complaing if you try grabbing a batch of images with differing heights and widths (or something like that). Need to resize them all to square dimensions to avoid this error.
4. Since we normalize the validation images during training, we can't visualize them unless we un-normalize them. I made a function to un-normalize a batch for visualization purposes. I figure we can just visualize the same batch during training to see how our captioning is changing.

#