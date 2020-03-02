# CSE253_PA4 Image Captioning

## Running the code

We would recommend starting by adding the following files to a new directory (on dsmlp server)

__STATIC FILES__<br>
TrainImageIds.csv<br>
TestImageIds.csv<br>
TheBestNapster.py<br>
data_loader.py<br>

__RUNNABLE FILES__
get_data.py<br>
word_holder.py<br>
subset_data.py<br>
resize.py<br>
train.py<br>
test.py<br>

Then running the runnable files in this order and manner:

```bash
python get_data.py
python word_holder.py
python subset_data.py
python resize.py
python train.py --run_id test_1
python test.py --run_id test_1
```

## File descriptions
__get_data.py__ -- script to retrieve data (currently specific to data being on dsmlp-login.ucsd.edu server. To see usage type python get_data.py --help on command line

__word_holder.py__ -- builds a vocabulary object that will be used in training and testing. To see usage, type python word_holder.py --help on the command line

__subset_data.py__ -- filters the list of caption ids to include only those that will actually be used during training and testing (basically filters based on if the file is in the training or test image folder or not). Saves this list of filtered ids that can be used in data loading. To see usage type python subset_data.py --help on command line

__resize.py__ -- takes a maximal center crop, and then resizes to a passed in value. This keeps the aspect ratio of the images the same. Performs this operation on both training and test set images and saves into a resize directory

__train.py__ -- script to train the model on training images. To see usage type python train.py --help on command line

__test.py__ -- script to calculate BLEU scores on test images for the passed in encoder and decoder. Use python test.py --help at command line for full set of options.

#### NOTE:
We are happy to provide the models used to generate the figures and results in our report. They can be passed into the test.py script.
