# CSE253_PA4
PA4 Image Captioning

train.py -- script to train the model on training images. To see usage type python train.py --help on command line

get_data.py -- script to retrieve data (currently specific to data being on dsmlp-login.ucsd.edu server. To see usage type python get_data.py --help on command line

dependencies.txt -- Run pip install dependencies.txt from command line to make sure you have all the necessary packages installed to run to train and test the model

subset_data.py -- filters the list of caption ids to include only those that will actually be used during training (basically filters based on if the file is in the training image folder or not). Saves this list of filtered ids that can be used in data loading


# TODO

- [ ] Function for validation split
- [ ] Skeleton for training loop
- [ ] Function (skeleton of one at least) for validation call after each train and at baseline