# CSE253_PA4 Image Captioning

## Running the code

I would recommend starting by adding the following files to a new directory (on dsmlp server)

__STATIC FILES__
TrainImageIds.csv
TestImageIds.csv
TheRealNapster.py
data_loader.py

__RUNNABLE FILES__
get_data.py
word_holder.py
subset_data.py
resize.py
train.py
test.py

Then running the runnable files in this order and manner:

```bash
python get_data.py
python word_holder.py
python subset_data.py
python resize.py
python train.py --run_id test_1
python test.py (not yet tested)
```

## File descriptions
  
train.py -- script to train the model on training images. To see usage type python train.py --help on command line

get_data.py -- script to retrieve data (currently specific to data being on dsmlp-login.ucsd.edu server. To see usage type python get_data.py --help on command line

dependencies.txt -- Run pip install dependencies.txt from command line to make sure you have all the necessary packages installed to run to train and test the model

subset_data.py -- filters the list of caption ids to include only those that will actually be used during training (basically filters based on if the file is in the training image folder or not). Saves this list of filtered ids that can be used in data loading. To see usage type python subset_data.py --help on command line

## TODO

- [ ] Add BLEU scoring to val and train
- [ ] Caption generation for validation set
