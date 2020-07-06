"""
1. load old data status, for example, mean, std, label, etc, input, output, 
those information will be saved from a dict from previous run.

2. new model without data normalization

3. modify new data std, mean

4. new normalization

5. embed old model into the new model to 
    5.1 the loss fcn.
    5.2 other layers
    5.3 other transfer learning? just some layers?

6. train the model

Note:
    1. mechanical data should be special case for just pre_process data. 
    No need to mess up with other data.
"""

def model_is_kbnn(config):
    """ 
    Based on the config file to determine if the model is a KBNN or not.
    return: True    if is KBNN
            False   if not
    """
    try:
        KBNN_flag = (config['KBNN']['LabelShiftingModels'] != '')
    except:
        KBNN_flag = False
        pass

    # the other type of KBNN is not coded
    # EmbeddingModels = 
    return KBNN_flag

def prepare_kbnn(config, dataset):
    """
    read index and frame
    shift_labels
    """
    import ddmms.ml_preprocess.read_csv_fields as read_csv_fields

    data_file = config['TEST']['DataFile']

    ## index and frames are used to match the CNN training info
    # index was the first try, but it turns out that we should use the base vtu file to predict the base free energy function.
    try:
        raw_dataset_index = read_csv_fields(data_file, ['index'])
        dataset_index = raw_dataset_index.copy()
    except:
        print("***ERR** in loading the index data. Will be neglected!!!")
        dataset_index = None
        pass

    # frame is the final choice. Rerun the collect data script to get new dataset if needed.
    try:
        raw_dataset_frame = read_csv_fields(data_file, ['frame'])
        dataset_frame = raw_dataset_frame.copy()
    except:
        print("***ERR** in loading the frame data. Will be neglected!!!")
        dataset_frame = None
        pass
    #----------NN label shift------------------
    import ddmms.models.KBNN as KBNN
    KBNN.shift_labels(config, dataset, dataset_index, dataset_frame,
                      data_file)
    
