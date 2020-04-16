from utils.config import Config
import model
import dataset
import logging
import dataset.NIH_pretrain_dataset as dpd
from model.NIH_model import create_NASnet_multioutupt, create_rewritten_complex_joint_multioutput,create_VGG_multioutput,create_resnet_multioutput,create_bigger_kernel_multioutput,create_densenet_multioutput,create_Xception_multioutput, create_rewritten_elu
from train.pretrain_NIH import pretrain_NIH_chest
from tensorflow.keras.models import load_model
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)

if __name__ == '__main__':

    # set up configuration
    configuration = Config()
    # after this logging.info() will print save to a specific file (global)!
    logging.info("configuration loaded")
    
    # prepare data
    logging.info("preparing train dataset")

    dataset = dpd.pretrain_dataset_NIH_chest(configuration)
    chest_dataset, chest_dataset_val = dataset.initialize_pipeline()

    logging.info("datasets prepared")

    for model_constr,name in zip([create_rewritten_elu,],["NIH_rewritten_elu_a0.1"]):
        model = model_constr(configuration)

        model.summary()

        logging.info("model prepared")
        # train
        logging.info("starting training")
        pretrain_NIH_chest(model,chest_dataset,chest_dataset_val,configuration,name,epochs=101)


# create_densenet_multioutput
# "NIH_densenet",
# create_rewritten_complex_joint_multioutput, create_VGG_multioutput, create_resnet_multioutput, create_bigger_kernel_multioutput, create_densenet_multioutput,create_Xception_multioutput, create_NASnet_multioutupt

# "NIH_rewritten","NIH_VGG", "NIH_resnet_moredense","NIH_bigger_kernel", "NIH_densenet","NIH_Xception","NIH_NASnet"