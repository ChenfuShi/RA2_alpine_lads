########################################




########################################


# pretrain model for joint recognition on faces.   just try a resnet model as well, to see
# train landmarks on joints. use one and then the other to get max power
# predict landmarks

# pretrain model for joint classification - pretrain on chest, then pretrain on joints from RSNA bone age to predict age and sex. only age > 100 months

# get weights from pretrains


# use predicted landmarks to get images for individual joints(splits up in like 2-3 models for hands and 1-2 models for feet)*2 for erosion and narrowing


# train on individual joints


# send model to tester to predict test dataset

# get landmarks with model, then the testdataset that already has landmarks will use those. all of these do not need to be tf.dataset. and won't be