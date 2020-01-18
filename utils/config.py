########################################




########################################

import logging
import datetime


class Config:
    """
    Class containing the parameters 
    """

    # data directories

    pretrain_location = ""

    train_location = "data/train"

    val_location = ""

    test_location = ""

    output_dir = "logs"

    # move files 
    move_files = False
    move_location = "~/localscratch/RA_challenge_scratch"

    # other options
    CPU_threads = 8


    # Create logging file. prints both to screen and to log file
    cur_date = datetime.datetime.now()
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
        handlers=[
        logging.FileHandler("{0}/{1}.log".format(output_dir, f"{cur_date.year}-{cur_date.month}-{cur_date.day}_{cur_date.hour}.{cur_date.minute}.{cur_date.second}"), mode="a"),
        logging.StreamHandler()
    ]
    )    