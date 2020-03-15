from utils.config import Config
import model
import dataset
import logging
import dataset.NIH_pretrain_dataset as dpd
from model.NIH_model_hyperparameters import *
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

    constructors = [create_densenet_multioutput_A, create_densenet_multioutput_B,
    create_densenet_multioutput_C, create_densenet_multioutput_D,
    create_densenet_multioutput_E, create_densenet_multioutput_F, 
    create_densenet_multioutput_G, create_densenet_multioutput_H]
    
    names = ["NIH_hyperaram_A","NIH_hyperaram_B","NIH_hyperaram_C","NIH_hyperaram_D",
    "NIH_hyperaram_E","NIH_hyperaram_F","NIH_hyperaram_G","NIH_hyperaram_H"]

    for model_constr,name in zip(constructors, names):   
        model = model_constr(configuration)

        #create_bigger_kernel_multioutput 

        model.summary()
        # check if there is weights to load
        logging.info("model prepared")
        # train
        logging.info("starting training")
        pretrain_NIH_chest(model,chest_dataset,chest_dataset_val,configuration,name,epochs=10)


