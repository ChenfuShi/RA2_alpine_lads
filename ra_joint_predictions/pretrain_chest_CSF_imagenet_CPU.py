from utils.config import Config
import model
import dataset
import logging
import dataset.NIH_pretrain_dataset as dpd
from model.NIH_model import create_VGG_multioutput_imagenet,create_resnet_multioutput_imagenet,create_densenet_multioutput_imagenet,create_NASnet_multioutupt_imagenet, create_Xception_multioutput
import tensorflow as tf
from train.pretrain_NIH import pretrain_NIH_chest
from tensorflow.keras.models import load_model

if __name__ == '__main__':

    # test the different image sizes imagenet
    configuration = Config()
    tf.config.threading.set_intra_op_parallelism_threads(24)
    tf.config.threading.set_inter_op_parallelism_threads(24)
    configuration.img_height = 299
    configuration.img_width = 299
    
    dataset = dpd.pretrain_dataset_NIH_chest(configuration)
    chest_dataset, chest_dataset_val = dataset.initialize_pipeline(imagenet = True)

    model = create_Xception_multioutput(configuration)

    model.summary()
    # check if there is weights to load
    logging.info("model prepared")
    # train
    logging.info("starting training")
    pretrain_NIH_chest(model,chest_dataset,chest_dataset_val,configuration,"NIH_Xception_imagenet",epochs=10)
