########################################




########################################





class Config:
    """
    Class containing the parameters 
    """

    # data directories

    pretrain_location = ""

    train_location = "data/train"

    val_location = ""

    test_location = ""

    # move files 
    move_files = False
    move_location = "~/localscratch/RA_challenge_scratch"

    # other options
    CPU_threads = 8