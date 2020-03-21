from utils.config import Config
import model
import dataset
import logging
import dataset.NIH_pretrain_dataset as dpd
from model.NIH_model import create_VGG_multioutput_imagenet,create_resnet_multioutput_imagenet,create_densenet_multioutput_imagenet,create_NASnet_multioutupt_imagenet, create_Xception_multioutput_imagenet

from train.pretrain_NIH import pretrain_NIH_chest
from tensorflow.keras.models import load_model

if __name__ == '__main__':

    # set up configuration
    configuration = Config()
    # after this logging.info() will print save to a specific file (global)!
    logging.info("configuration loaded")
    
    # prepare data
    logging.info("preparing train dataset")

    dataset = dpd.pretrain_dataset_NIH_chest(configuration)
    chest_dataset, chest_dataset_val = dataset.initialize_pipeline(imagenet = True)

    logging.info("datasets prepared")


    # create_VGG_multioutput_imagenet,create_resnet_multioutput_imagenet,create_densenet_multioutput_imagenet,"NIH_resnet_imagenet","NIH_densenet_imagenet",
    # "NIH_VGG_imagenet_max",
    for model_constr,name in zip([create_densenet_multioutput_imagenet],["NIH_densenet_imagenet"]):
        model = model_constr(configuration)

        #create_bigger_kernel_multioutput 

        model.summary()
        # check if there is weights to load
        logging.info("model prepared")
        # train
        logging.info("starting training")
        pretrain_NIH_chest(model,chest_dataset,chest_dataset_val,configuration,name,epochs=101)


    # test the different image sizes imagenet
    configuration = Config()

    configuration.img_height = 299
    configuration.img_width = 299
    configuration.batch_size = 32
    dataset = dpd.pretrain_dataset_NIH_chest(configuration)
    chest_dataset, chest_dataset_val = dataset.initialize_pipeline(imagenet = True)

    model = create_Xception_multioutput_imagenet(configuration)

    model.summary()
    # check if there is weights to load
    logging.info("model prepared")
    # train
    logging.info("starting training")
    pretrain_NIH_chest(model,chest_dataset,chest_dataset_val,configuration,"NIH_Xception_imagenet",epochs=101)