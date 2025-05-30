# Parameters used in the feature extraction, neural network model, and training the SELDnet can be changed here.
#
# Ideally, do not change the values of the default parameters. Create separate cases with unique <task-id> as seen in
# the code below (if-else loop) and use them. This way you can easily reproduce a configuration on a later time.


def get_params(argv='1', train=False, verbose=False):
    print("SET: {}".format(argv))
    # ########### default parameters ##############
    params = dict(
        quick_test=False,  # To do quick test. Trains/test on small subset of dataset, and # of epochs

        finetune_mode=True,  # Finetune on existing model, requires the pretrained model path set - pretrained_model_weights
        pretrained_model_weights='3_1_dev_split0_multiaccdoa_foa_model.h5',

        # INPUT PATH
        # dataset_dir='DCASE2020_SELD_dataset/',  # Base folder containing the foa/mic and metadata folders
        dataset_dir='',
        eval_dataset_dir = '/scratch/ci411/SELD/seld_datasets/STARSS23',

        # OUTPUT PATHS
        # feat_label_dir='DCASE2020_SELD_dataset/feat_label_hnet/',  # Directory to dump extracted features and labels
        feat_label_dir='/scratch/ci411/SELD/seld_features/exp0_baseline_features',

        model_dir='/scratch/ci411/SELD/models', 
        # Dumps the trained models and training curves in this folder
        dcase_output_dir='/scratch/ci411/SELD/results',  # recording-wise results are dumped in this path.
        unique_name = None,

        project = "SELD_Baselines_01",

        # DATASET LOADING PARAMETERS
        mode='dev',  # 'dev' - development or 'eval' - evaluation dataset
        dataset='foa',  # 'foa' - ambisonic or 'mic' - microphone signals

        train_splits = [[1,2,3]],
        val_splits = [[4]],
        test_splits = [[4]],

        unique_classes= 13,

        # FEATURE PARAMS
        fs=24000,
        hop_len_s=0.02,
        label_hop_len_s=0.1,
        max_audio_len_s=60,
        nb_mel_bins=64,

        use_salsalite=False,  # Used for MIC dataset only. If true use salsalite features, else use GCC features
        fmin_doa_salsalite=50,
        fmax_doa_salsalite=2000,
        fmax_spectra_salsalite=9000,

        # MODEL TYPE
        modality='audio',  # 'audio' or 'audio_visual'
        multi_accdoa=True,  # False - Single-ACCDOA or True - Multi-ACCDOA
        thresh_unify=15,    # Required for Multi-ACCDOA only. Threshold of unification for inference in degrees.

        # DNN MODEL PARAMETERS
        label_sequence_length=50,    # Feature sequence length
        batch_size=128,              # Batch size
        dropout_rate=0.05,           # Dropout rate, constant for all layers
        nb_cnn2d_filt=64,           # Number of CNN nodes, constant for each layer
        f_pool_size=[4, 4, 2],      # CNN frequency pooling, length of list = number of CNN layers, list value = pooling per layer

        nb_heads=8,
        nb_self_attn_layers=2,
        nb_transformer_layers=2,

        nb_rnn_layers=2,
        rnn_size=128,

        nb_fnn_layers=1,
        fnn_size=128,  # FNN contents, length of list = number of layers, list value = number of nodes

        nb_epochs=1000,  # Train for maximum epochs

        lr_type = 'static', #['static', 'switch_cyclic', 'scheduled']
        
        lr=1e-3,
        switch_epoch = 500,
        min_lr = 1e-4,
        max_lr = 1e-3,

        lr_schedule = [500,750,900],
        lr_gamma = 0.1,

        # METRIC
        average='macro',                 # Supports 'micro': sample-wise average and 'macro': class-wise average,
        segment_based_metrics=False,     # If True, uses segment-based metrics, else uses frame-based metrics
        evaluate_distance=True,          # If True, computes distance errors and apply distance threshold to the detections
        lad_doa_thresh=20,               # DOA error threshold for computing the detection metrics
        lad_dist_thresh=float('inf'),    # Absolute distance error threshold for computing the detection metrics
        lad_reldist_thresh=float('1'),  # Relative distance error threshold for computing the detection metrics
    )

    # ########### User defined parameters ##############
    if argv == '1':
        print("USING DEFAULT PARAMETERS\n")

    elif argv == '3':
        print("FOA + multi ACCDOA\n")

    elif argv == '3_starss':
        print("STARSS\n")
        params['dataset_dir'] = "/scratch/ci411/SELD/seld_datasets/STARSS23"
        params['feat_label_dir'] = "/scratch/ci411/SELD/seld_features/starss_features"

    elif argv == '3_ssbaseline':
        print("Spatial_Scaper Baseline")
        params['dataset_dir'] = "/scratch/ci411/SELD/seld_datasets/SSBaseline"
        params['feat_label_dir'] = "/scratch/ci411/SELD/seld_features/ssbaseline_features"
    
    elif argv == '3_ssbaseline_revert':
        print("Spatial_Scaper Baseline")
        params['dataset_dir'] = "/scratch/ci411/SELD/seld_datasets/SSBaseline_revert"
        params['feat_label_dir'] = "/scratch/ci411/SELD/seld_features/ssbaseline-revert_features"
    
    elif argv == '3_nafbaseline':
        print("NAF Baseline\n")
        params['dataset_dir'] = "/scratch/ci411/SELD/seld_datasets/NAFBaseline"
        params['feat_label_dir'] = "/scratch/ci411/SELD/seld_features/naf_baseline_features"

    elif argv == '3_ismbaseline':
        print("ISM Baseline\n")
        params['dataset_dir'] = "/scratch/ci411/SELD/seld_datasets/ism_rescale"
        params['feat_label_dir'] = "/scratch/ci411/SELD/seld_features/ism_baseline_features"

    elif argv == '3_ismbaseline_revert':
        print("ISM Baseline\n")
        params['dataset_dir'] = "/scratch/ci411/SELD/seld_datasets/ism_revert2"
        params['feat_label_dir'] = "/scratch/ci411/SELD/seld_features/ism_revert2_features"

    elif argv == '3_ism_yw':
        print("ISM Baseline Yi's features\n")
        params['dataset_dir'] = "/scratch/ci411/SELD/seld_datasets/ism_yw"
        params['feat_label_dir'] = "/scratch/ci411/SELD/seld_features/ism_yw_features"

    elif argv == '3_exp0s':
        print("STARSS on STARSS\n")
        params['feat_label_dir'] = "/scratch/ci411/SELD/seld_features/exp0_baseline_features"
        params['train_splits']  = [[3]]
    
    elif argv == '3_exp0':
        print("Baseline Data\n")
        params['feat_label_dir'] = "/scratch/ci411/SELD/seld_features/exp0_baseline_features"
        
    elif argv == '3_exp1':
        print("SS Regen Test\n")
        params['feat_label_dir'] = "/scratch/ci411/SELD/seld_features/exp1_regen_features"

    elif argv == '3_exp1r':
        print("SS Revert Test SC\n")
        params['feat_label_dir'] = "/scratch/ci411/SELD/seld_features/exp1r_revert_features"
        
    elif argv == '3_exp2':
        print("NAF Baseline\n")
        params['feat_label_dir'] = "/scratch/ci411/SELD/seld_features/exp2_naf_features"

    elif argv == '3_exp3':
        print("ISM Baseline\n")
        params['feat_label_dir'] = "/scratch/ci411/SELD/seld_features/exp3_ism_features"
        
    elif argv == '3_exp3r':
        print("ISM Baseline REVERT\n")
        params['feat_label_dir'] = "/scratch/ci411/SELD/seld_features/exp3r_ism-revert_features"
    
    elif argv == '3_exp3r2':
        print("ISM Baseline REVERT\n")
        params['feat_label_dir'] = "/scratch/ci411/SELD/seld_features/exp3r2_ism-revert2_features"
        
    elif argv == '3_exp3yw':
        print("ISM Baseline Yi's Soundscape\n")
        params['feat_label_dir'] = "/scratch/ci411/SELD/seld_features/exp3yw_ism_features"


    elif argv[0] == 'd':
        print("FOA + multi ACCDOA\n")
        params['project'] = "Density_02"
        params['dataset_dir'] = f"/scratch/ci411/SELD/seld_datasets/DS_ISM_revert/{argv}"
        params['train_splits']  = [[1]]
        params['val_splits']  = [[2]]
        params['test_splits']  = [[2]]
        params['eval_dataset_dir'] = "/scratch/ci411/SELD/seld_datasets/DS_ISM_revert/d_1"
        if train:
            params['feat_label_dir'] = f"/scratch/ci411/SELD/seld_features/exp4_density_scaling/{argv}"
        else:
            params['feat_label_dir'] = f"/scratch/ci411/SELD/seld_features/density_features/{argv}"

    elif argv[0] == 'r':
        print(f"Reflection Order {argv}\n")
        params['dataset_dir'] = f"/scratch/ci411/SELD/seld_datasets/ism_yw_ro/{argv}"
        params['project'] = "Reflection_Order_03"
        #params['train_splits']  = [[1]]
        #params['val_splits']  = [[2]]
        #params['test_splits']  = [[2]]
        #params['eval_dataset_dir'] = "/scratch/ci411/SELD/seld_datasets/RO_ISM/r_20"
        if train:
            params['feat_label_dir'] = f"/scratch/ci411/SELD/seld_features/exp5yw_reflection_order/{argv}"
        else:    
            params['feat_label_dir'] = f"/scratch/ci411/SELD/seld_features/ism_yw_ro/{argv}"


    elif argv == '999':
        print("QUICK TEST MODE\n")
        params['quick_test'] = True

    else:
        print('ERROR: unknown argument {}'.format(argv))
        exit()

    feature_label_resolution = int(params['label_hop_len_s'] // params['hop_len_s'])
    params['feature_sequence_length'] = params['label_sequence_length'] * feature_label_resolution
    params['t_pool_size'] = [feature_label_resolution, 1, 1]  # CNN time pooling
    params['patience'] = int(params['nb_epochs'])  # Stop training if patience is reached
    params['model_dir'] = params['model_dir'] + '_' + params['modality']
    params['dcase_output_dir'] = params['dcase_output_dir'] + '_' + params['modality']
      

    if verbose:
        for key, value in params.items():
            print("\t{}: {}".format(key, value))
    return params
