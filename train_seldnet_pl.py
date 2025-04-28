#
# A wrapper script that trains the SELDnet. The training stops when the early stopping metric - SELD error stops improving.
#

import os
import sys
import numpy as np
import matplotlib.pyplot as plot
import cls_feature_class
import cls_data_generator
import parameters
import time
from time import gmtime, strftime
import torch
import torch.nn as nn
import torch.optim as optim
plot.switch_backend('agg')
from IPython import embed
from cls_compute_seld_results import ComputeSELDResults, reshape_3Dto2D
from SELD_evaluation_metrics import distance_between_cartesian_coordinates
from seldnet_pl import SeldModelPL, DataLoaderShuffleCallback

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger

def main(argv):
    """
    Main wrapper for training sound event localization and detection network.

    :param argv: expects two optional inputs.
        first input: task_id - (optional) To chose the system configuration in parameters.py.
                                (default) 1 - uses default parameters
        second input: unique_name - (optional) Name identifying the run, all outputs will be labeled

    """
    print(argv)
    if len(argv) != 3:
        print('\n\n')
        print('-------------------------------------------------------------------------------------------------------')
        print('The code expected two optional inputs')
        print('\t>> python seld.py <task-id> <job-id>')
        print('\t\t<task-id> is used to choose the user-defined parameter set from parameter.py')
        print('Using default inputs for now')
        print('\t\t<job-id> is a unique identifier which is used for output filenames (models, training plots). '
              'You can use any number or string for this.')
        print('-------------------------------------------------------------------------------------------------------')
        print('\n\n')

    # use parameter set defined by user
    task_id = '1' if len(argv) < 2 else argv[1]
    params = parameters.get_params(task_id, train=True)

    unique_name = params['unique_name'] if len(argv) < 3 else argv[-1]

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Torch is using {device}")
    
    # Training setup
    train_splits = params['train_splits']
    val_splits = params['val_splits']
    test_splits = params['test_splits']

    if params['mode'] == 'dev':
        for split_cnt, split in enumerate(test_splits):
            print('\n\n---------------------------------------------------------------------------------------------------')
            print('------------------------------------      SPLIT {}   -----------------------------------------------'.format(split))
            print('---------------------------------------------------------------------------------------------------')

            # Unique name for the run
            loc_feat = params['dataset']
            loc_output = 'multiaccdoa'

            cls_feature_class.create_folder(params['model_dir'])

            if unique_name is None:
                unique_name = '{}_{}_{}_split{}_{}_{}'.format(
                    task_id, job_id, params['mode'], split_cnt, loc_output, loc_feat
                )
            
            model_name = '{}_model.h5'.format(os.path.join(params['model_dir'], unique_name))
            print("unique_name: {}\n".format(unique_name))

            # Load train and validation data
            print(f'Loading training dataset from {params['feat_label_dir']}:')
            data_gen_train = cls_data_generator.DataGenerator(
                params=params, split=train_splits[split_cnt], device=device,
            )

            print(f'Loading validation dataset from {params['feat_label_dir']}:')
            data_gen_val = cls_data_generator.DataGenerator(
                params=params, split=val_splits[split_cnt], device=device, shuffle=False, per_file=True
            )

            # Collect i/o data size and load model configuration
            print('Loading model spec...')
            data_in, data_out = data_gen_train.get_data_sizes()
            model = SeldModelPL(data_in, data_out, params).to(device)


            print('---------------- SELD-net -------------------')
            print('FEATURES:\n\tdata_in: {}\n\tdata_out: {}\n'.format(data_in, data_out))
            print('MODEL:\n\tdropout_rate: {}\n\tCNN: nb_cnn_filt: {}, f_pool_size{}, t_pool_size{}\n, rnn_size: {}\n, nb_attention_blocks: {}\n, fnn_size: {}\n'.format(
                params['dropout_rate'], params['nb_cnn2d_filt'], params['f_pool_size'], params['t_pool_size'], params['rnn_size'], params['nb_self_attn_layers'],
                params['fnn_size']))
            print(model)

            nb_epoch = 2 if params['quick_test'] else params['nb_epochs']
            optimizer = optim.Adam(model.parameters(), lr=params['lr'])
           
            logger = WandbLogger(
                project=params["project"],
                name=unique_name
            )

            trainer = L.Trainer(
                                logger=logger,
                                max_epochs=nb_epoch,
                                devices="auto",
                                accelerator="auto",
                                log_every_n_steps=1,
                                check_val_every_n_epoch=10,
                                callbacks=[
                                    ModelCheckpoint(
                                        dirpath=os.path.join(params['model_dir'], unique_name),
                                        filename="best",
                                        monitor="val_SELDScore",
                                        auto_insert_metric_name=True,
                                        save_top_k=1 # save top two best models for this criteron
                                    ),
                                    DataLoaderShuffleCallback()
                                ],
                                )

            trainer.fit(
                model=model,
                train_dataloaders=data_gen_train,
                val_dataloaders=data_gen_val,
            )

            # ---------------------------------------------------------------------
            # Evaluate on unseen test data
            # ---------------------------------------------------------------------

            print('Loading unseen test dataset:')
            data_gen_test = cls_data_generator.DataGenerator(
                params=params, split=test_splits[split_cnt], device=device, shuffle=False, per_file=True
            )

            trainer.test(model=model,
                        dataloaders=data_gen_test,
                        ckpt_path=os.path.join(params['model_dir'], unique_name, 'best.ckpt'),
                        verbose=True)

    if params['mode'] == 'eval':

        print('Loading evaluation dataset:')
        data_gen_eval = cls_data_generator.DataGenerator(
            params=params, device=device, shuffle=False, per_file=True, is_eval=True)


        data_in, data_out = data_gen_eval.get_data_sizes()
        model = SeldModelPL(data_in, data_out, params).to(device)

        trainer.test(model=model, 
                     dataloaders=data_gen+test,
                     ckpt_path=os.path.join(params['model_dir'], unique_name, 'best.ckpt'),
                     verbose=True)
        
if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)