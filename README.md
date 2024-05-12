# HemorrhageDetection
Goal: Find the most efficient way to optimize the classification of Brain Hemorrhage using ML

Authors: Kevin Ciardelli, Alicia Doung, Saskriti Neupane

Feel free to look at the included Report and presentation for more detail on the process

# Runnable scripts/commands: what are the main commands that you run to get the results

As seen within DataDownload.ipynb we get access to our data to test using: 

fn = get_image_files('rsna-data/train_jpg/train_jpg/')

From here customizing csv files to correspond to what type of data we want to test by filtering:

file_list = os.listdir('train_jpg/train_jpg')
Filter the DataFrame to include only files that exist in the directory
labels = labels[labels['ID'].isin(file_list)]

Then we can access the individual types:
subdural_data = batch_multi_labels[batch_multi_labels['multi'].str.contains('subdural')]
display(subdural_data)

After aquiring the data we want to test on and corresponding it to the correct csv file we can run our models:

This prepares our batch size, traing set, validation set:
dls= ImageDataLoaders.from_df(labels, 'train_jpg/train_jpg/', bs = 32, seed = 42)

We can then choose a model:
learn_baseline = vision_learner(dls_batch, resnet18, metrics=accuracy_multi)

And tell how many epochs to run with an associated learning rate:
learn_baseline.fit_one_cycle(2,0.004365158267319202)

# Contribution per folder:

Data - Alicia, Saskriti

Baseline - Kevin

Resnet34 - Saskriti

Alexnet - Kevin

DenseNet - Alicia