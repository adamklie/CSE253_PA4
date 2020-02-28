# CSE253_PA4 Image Captioning

## Running the code

I would recommend starting by adding the following files to a new directory (on dsmlp server)

__STATIC FILES__
TrainImageIds.csv
TestImageIds.csv
TheBestNapster.py
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
python test.py --run_id test_1
```

## File descriptions
__get_data.py__ -- script to retrieve data (currently specific to data being on dsmlp-login.ucsd.edu server. To see usage type python get_data.py --help on command line

__word_holder.py__ --

__subset_data.py__ -- filters the list of caption ids to include only those that will actually be used during training (basically filters based on if the file is in the training image folder or not). Saves this list of filtered ids that can be used in data loading. To see usage type python subset_data.py --help on command line

__resize.py__

__train.py__ -- script to train the model on training images. To see usage type python train.py --help on command line

__test.py__


<font color="red"> __dependencies.txt__ -- Run pip install dependencies.txt from command line to make sure you have all the necessary packages installed to run to train and test the model <\font>



## TODO

- [ ] Hyperparameter search and recording of different scores
- [ ] Test pretrained embeddings/rnn/stochastic on BLEU scoring to be sure we are getting comparable results
- [ ] Re-read assignment to make sure we have implemented everything
- [ ] Run models for long time for final results