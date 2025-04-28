import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR, MultiStepLR
import math
from IPython import embed
from time import gmtime, strftime
import random

import yaml

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger

from seldnet_model import MSELoss_ADPIT, ConvBlock
import cls_feature_class
from cls_compute_seld_results import ComputeSELDResults, reshape_3Dto2D
from SELD_evaluation_metrics import distance_between_cartesian_coordinates

def get_accdoa_labels(accdoa_in, nb_classes):
    x, y, z = accdoa_in[:, :, :nb_classes], accdoa_in[:, :, nb_classes:2*nb_classes], accdoa_in[:, :, 2*nb_classes:]
    sed = np.sqrt(x**2 + y**2 + z**2) > 0.5
      
    return sed, accdoa_in


def get_multi_accdoa_labels(accdoa_in, nb_classes):
    """
    Args:
        accdoa_in:  [batch_size, frames, num_track*num_axis*num_class=3*3*12]
        nb_classes: scalar
    Return:
        sedX:       [batch_size, frames, num_class=12]
        doaX:       [batch_size, frames, num_axis*num_class=3*12]
    """
    x0, y0, z0 = accdoa_in[:, :, :1*nb_classes], accdoa_in[:, :, 1*nb_classes:2*nb_classes], accdoa_in[:, :, 2*nb_classes:3*nb_classes]
    dist0 = accdoa_in[:, :, 3*nb_classes:4*nb_classes]
    dist0[dist0 < 0.] = 0.
    sed0 = np.sqrt(x0**2 + y0**2 + z0**2) > 0.5
    doa0 = accdoa_in[:, :, :3*nb_classes]

    x1, y1, z1 = accdoa_in[:, :, 4*nb_classes:5*nb_classes], accdoa_in[:, :, 5*nb_classes:6*nb_classes], accdoa_in[:, :, 6*nb_classes:7*nb_classes]
    dist1 = accdoa_in[:, :, 7*nb_classes:8*nb_classes]
    dist1[dist1<0.] = 0.
    sed1 = np.sqrt(x1**2 + y1**2 + z1**2) > 0.5
    doa1 = accdoa_in[:, :, 4*nb_classes: 7*nb_classes]

    x2, y2, z2 = accdoa_in[:, :, 8*nb_classes:9*nb_classes], accdoa_in[:, :, 9*nb_classes:10*nb_classes], accdoa_in[:, :, 10*nb_classes:11*nb_classes]
    dist2 = accdoa_in[:, :, 11*nb_classes:]
    dist2[dist2<0.] = 0.
    sed2 = np.sqrt(x2**2 + y2**2 + z2**2) > 0.5
    doa2 = accdoa_in[:, :, 8*nb_classes:11*nb_classes]

    return sed0, doa0, dist0, sed1, doa1, dist1, sed2, doa2, dist2


def determine_similar_location(sed_pred0, sed_pred1, doa_pred0, doa_pred1, class_cnt, thresh_unify, nb_classes):
    if (sed_pred0 == 1) and (sed_pred1 == 1):
        if distance_between_cartesian_coordinates(doa_pred0[class_cnt], doa_pred0[class_cnt+1*nb_classes], doa_pred0[class_cnt+2*nb_classes],
                                                  doa_pred1[class_cnt], doa_pred1[class_cnt+1*nb_classes], doa_pred1[class_cnt+2*nb_classes]) < thresh_unify:
            return 1
        else:
            return 0
    else:
        return 0


class SeldModelPL(L.LightningModule):
    def __init__(self, in_feat_shape, out_shape, params):
        super().__init__()
        self.params = params
        # Initialize model from other file
        if params['multi_accdoa'] is True:
            self.loss_fn = MSELoss_ADPIT()
        else:
            self.loss_fn = nn.MSELoss()

        self.score_obj = ComputeSELDResults(params)

        if "unique_name" in params:
            unique_name = params["unique_name"]
        else:
            unique_name = "NO-NAME"
        
        self.output_val_folder = os.path.join(params['dcase_output_dir'], '{}_{}_val'.format(unique_name, strftime("%Y%m%d%H%M%S", gmtime())))
        self.output_test_folder = os.path.join(params['dcase_output_dir'], '{}_{}_test'.format(unique_name, strftime("%Y%m%d%H%M%S", gmtime())))

        cls_feature_class.delete_and_create_folder(self.output_val_folder)
        cls_feature_class.delete_and_create_folder(self.output_test_folder)

        print('Dumping recording-wise val results in: {}'.format(self.output_val_folder))
        print('Dumping recording-wise test results in: {}'.format(self.output_test_folder))
        
        self.nb_classes = params['unique_classes']
        self.thresh_unify = params['thresh_unify']
        self.params=params
        self.conv_block_list = nn.ModuleList()
        if len(params['f_pool_size']):
            for conv_cnt in range(len(params['f_pool_size'])):
                self.conv_block_list.append(ConvBlock(in_channels=params['nb_cnn2d_filt'] if conv_cnt else in_feat_shape[1], out_channels=params['nb_cnn2d_filt']))
                self.conv_block_list.append(nn.MaxPool2d((params['t_pool_size'][conv_cnt], params['f_pool_size'][conv_cnt])))
                self.conv_block_list.append(nn.Dropout2d(p=params['dropout_rate']))

        self.gru_input_dim = params['nb_cnn2d_filt'] * int(np.floor(in_feat_shape[-1] / np.prod(params['f_pool_size'])))
        self.gru = torch.nn.GRU(input_size=self.gru_input_dim, hidden_size=params['rnn_size'],
                                num_layers=params['nb_rnn_layers'], batch_first=True,
                                dropout=params['dropout_rate'], bidirectional=True)

        self.mhsa_block_list = nn.ModuleList()
        self.layer_norm_list = nn.ModuleList()
        for mhsa_cnt in range(params['nb_self_attn_layers']):
            self.mhsa_block_list.append(nn.MultiheadAttention(embed_dim=self.params['rnn_size'], num_heads=self.params['nb_heads'], dropout=self.params['dropout_rate'], batch_first=True))
            self.layer_norm_list.append(nn.LayerNorm(self.params['rnn_size']))

        self.fnn_list = torch.nn.ModuleList()
        if params['nb_fnn_layers']:
            for fc_cnt in range(params['nb_fnn_layers']):
                self.fnn_list.append(nn.Linear(params['fnn_size'] if fc_cnt else self.params['rnn_size'], params['fnn_size'], bias=True))
        self.fnn_list.append(nn.Linear(params['fnn_size'] if params['nb_fnn_layers'] else self.params['rnn_size'], out_shape[-1], bias=True))

        self.doa_act = nn.Tanh()
        self.dist_act = nn.ReLU()

        self._lr_type = params['lr_type'] #switch_cyclic, scheduled, or adaptive
        self._lr = params['lr']
        self._min_lr = params['min_lr']
        self._max_lr = params['max_lr']
        self._switch_epoch = params['switch_epoch']
        
        self._lr_schedule = params['lr_schedule']
        self._lr_gamma = params['lr_gamma']


        self._dataset_combination = '{}_{}'.format(params['dataset'], params['mode'])
        self._feat_label_dir = params['feat_label_dir']
        self._feat_dir = os.path.join(self._feat_label_dir,'{}_norm'.format(self._dataset_combination))

    def configure_optimizers(self):
        if self._lr_type == "static":
            optimizer = optim.Adam(self.parameters(), lr=self._lr)
            return optimizer
            
        elif self._lr_type == "switch_cyclic":
            if self.trainer.current_epoch < self._switch_epoch:
                optimizer = optim.Adam(self.parameters(), lr=self._lr)
                return optimizer
            else:
                optimizer = optim.Adam(self.parameters(), lr=self._lr) # Start with a base learning rate
                scheduler = {
                    'scheduler': CyclicLR(optimizer,
                                           base_lr=self._min_lr,
                                           max_lr=self._max_lr,
                                           step_size_up=500,
                                           step_size_down=500,
                                           mode='triangular'),
                    'interval': 'step'  # Update the learning rate after each training step
                }
                return {'optimizer': optimizer, 'lr_scheduler': scheduler}
                
        elif self._lr_type == "scheduled":
            optimizer = optim.Adam(self.parameters(), lr=self._lr)
            scheduler = MultiStepLR(optimizer, milestones=self._lr_schedule, gamma=self._lr_gamma)
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    def forward(self, x):
        """input: (batch_size, mic_channels, time_steps, mel_bins)"""
        for conv_cnt in range(len(self.conv_block_list)):
            x = self.conv_block_list[conv_cnt](x)

        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        (x, _) = self.gru(x)
        x = torch.tanh(x)
        x = x[:, :, x.shape[-1]//2:] * x[:, :, :x.shape[-1]//2]

        for mhsa_cnt in range(len(self.mhsa_block_list)):
            x_attn_in = x
            x, _ = self.mhsa_block_list[mhsa_cnt](x_attn_in, x_attn_in, x_attn_in)
            x = x + x_attn_in
            x = self.layer_norm_list[mhsa_cnt](x)

        for fnn_cnt in range(len(self.fnn_list) - 1):
            x = self.fnn_list[fnn_cnt](x)
        doa = self.fnn_list[-1](x)
        
        return doa

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.loss_fn(output, target)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        data, target, filename = batch 
        output = self(data)
        loss = self.loss_fn(output, target)
        self.log('val_loss_epoch', loss,  prog_bar=True, on_step=False, on_epoch=True)
        output_file = os.path.join(self.output_val_folder, filename.replace('.npy', '.csv'))
        self.write_seld_predictions(output, output_file)
        return loss

    def on_validation_epoch_end(self):
        ER, F, LE, dist_err, rel_dist_err, LR, seld_scr, classwise_val_scr = self.score_obj.get_SELD_Results(self.output_val_folder)
        self.log('val_ErrorRate', ER, on_step=False, on_epoch=True)
        self.log('val_FScore', F, on_step=False, on_epoch=True)
        self.log('val_LocalizationError', LE, on_step=False, on_epoch=True)
        self.log('val_DistanceError', dist_err, on_step=False, on_epoch=True)
        self.log('val_RelDistanceError', rel_dist_err, on_step=False, on_epoch=True)
        self.log('val_LocalizationRecall', LR, on_step=False, on_epoch=True)
        self.log('val_SELDScore', seld_scr, on_step=False, on_epoch=True)
    
    def test_step(self, batch, batch_idx):
        data, target, filename = batch 
        output = self(data)
        loss = self.loss_fn(output, target)
        self.log('test_loss_epoch', loss,  prog_bar=True, on_step=False, on_epoch=True)
        output_file = os.path.join(self.output_test_folder,filename.replace('.npy', '.csv'))
        
        self.write_seld_predictions(output, output_file)
        
    def on_test_epoch_end(self):
        ER, F, LE, dist_err, rel_dist_err, LR, seld_scr, classwise_val_scr = self.score_obj.get_SELD_Results(self.output_test_folder, is_jackknife=True)
        self.log('test_ErrorRate', ER, on_step=False, on_epoch=True)
        self.log('test_FScore', F, on_step=False, on_epoch=True)
        self.log('test_LocalizationError', LE, on_step=False, on_epoch=True)
        self.log('test_DistanceError', dist_err, on_step=False, on_epoch=True)
        self.log('test_RelDistanceError', rel_dist_err, on_step=False, on_epoch=True)
        self.log('test_LocalizationRecall', LR, on_step=False, on_epoch=True)
        self.log('test_SELDScore', seld_scr, on_step=False, on_epoch=True)
        
    def write_seld_predictions(self, output, output_file):
        sed_pred0, doa_pred0, dist_pred0, sed_pred1, doa_pred1, dist_pred1, sed_pred2, doa_pred2, dist_pred2 = get_multi_accdoa_labels(output.detach().cpu().numpy(), self.nb_classes)
        sed_pred0 = reshape_3Dto2D(sed_pred0)
        doa_pred0 = reshape_3Dto2D(doa_pred0)
        dist_pred0 = reshape_3Dto2D(dist_pred0)
        sed_pred1 = reshape_3Dto2D(sed_pred1)
        doa_pred1 = reshape_3Dto2D(doa_pred1)
        dist_pred1 = reshape_3Dto2D(dist_pred1)
        sed_pred2 = reshape_3Dto2D(sed_pred2)
        doa_pred2 = reshape_3Dto2D(doa_pred2)
        dist_pred2 = reshape_3Dto2D(dist_pred2)
        
        output_dict = {}
        for frame_cnt in range(sed_pred0.shape[0]):
            for class_cnt in range(sed_pred0.shape[1]):
                # determine whether track0 is similar to track1
                flag_0sim1 = determine_similar_location(sed_pred0[frame_cnt][class_cnt], sed_pred1[frame_cnt][class_cnt], doa_pred0[frame_cnt], doa_pred1[frame_cnt], class_cnt, self.thresh_unify, self.nb_classes)
                flag_1sim2 = determine_similar_location(sed_pred1[frame_cnt][class_cnt], sed_pred2[frame_cnt][class_cnt], doa_pred1[frame_cnt], doa_pred2[frame_cnt], class_cnt, self.thresh_unify, self.nb_classes)
                flag_2sim0 = determine_similar_location(sed_pred2[frame_cnt][class_cnt], sed_pred0[frame_cnt][class_cnt], doa_pred2[frame_cnt], doa_pred0[frame_cnt], class_cnt, self.thresh_unify, self.nb_classes)
                # unify or not unify according to flag
                if flag_0sim1 + flag_1sim2 + flag_2sim0 == 0:
                    if sed_pred0[frame_cnt][class_cnt]>0.5:
                        if frame_cnt not in output_dict:
                            output_dict[frame_cnt] = []
                        output_dict[frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt], doa_pred0[frame_cnt][class_cnt+self.nb_classes], doa_pred0[frame_cnt][class_cnt+2*self.nb_classes], dist_pred0[frame_cnt][class_cnt]])
                    if sed_pred1[frame_cnt][class_cnt]>0.5:
                        if frame_cnt not in output_dict:
                            output_dict[frame_cnt] = []
                        output_dict[frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt], doa_pred1[frame_cnt][class_cnt+self.nb_classes], doa_pred1[frame_cnt][class_cnt+2*self.nb_classes], dist_pred1[frame_cnt][class_cnt]])
                    if sed_pred2[frame_cnt][class_cnt]>0.5:
                        if frame_cnt not in output_dict:
                            output_dict[frame_cnt] = []
                        output_dict[frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt], doa_pred2[frame_cnt][class_cnt+self.nb_classes], doa_pred2[frame_cnt][class_cnt+2*self.nb_classes], dist_pred2[frame_cnt][class_cnt]])
                elif flag_0sim1 + flag_1sim2 + flag_2sim0 == 1:
                    if frame_cnt not in output_dict:
                        output_dict[frame_cnt] = []
                    if flag_0sim1:
                        if sed_pred2[frame_cnt][class_cnt]>0.5:
                            output_dict[frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt], doa_pred2[frame_cnt][class_cnt+self.nb_classes], doa_pred2[frame_cnt][class_cnt+2*self.nb_classes], dist_pred2[frame_cnt][class_cnt]])
                        doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt]) / 2
                        dist_pred_fc = (dist_pred0[frame_cnt] + dist_pred1[frame_cnt]) / 2
                        output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+self.nb_classes], doa_pred_fc[class_cnt+2*self.nb_classes], dist_pred_fc[class_cnt]])
                    elif flag_1sim2:
                        if sed_pred0[frame_cnt][class_cnt]>0.5:
                            output_dict[frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt], doa_pred0[frame_cnt][class_cnt+self.nb_classes], doa_pred0[frame_cnt][class_cnt+2*self.nb_classes], dist_pred0[frame_cnt][class_cnt]])
                        doa_pred_fc = (doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 2
                        dist_pred_fc = (dist_pred1[frame_cnt] + dist_pred2[frame_cnt]) / 2
                        output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+self.nb_classes], doa_pred_fc[class_cnt+2*self.nb_classes], dist_pred_fc[class_cnt]])
                    elif flag_2sim0:
                        if sed_pred1[frame_cnt][class_cnt]>0.5:
                            output_dict[frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt], doa_pred1[frame_cnt][class_cnt+self.nb_classes], doa_pred1[frame_cnt][class_cnt+2*self.nb_classes], dist_pred1[frame_cnt][class_cnt]])
                        doa_pred_fc = (doa_pred2[frame_cnt] + doa_pred0[frame_cnt]) / 2
                        dist_pred_fc = (dist_pred2[frame_cnt] + dist_pred0[frame_cnt]) / 2
                        output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+self.nb_classes], doa_pred_fc[class_cnt+2*self.nb_classes], dist_pred_fc[class_cnt]])
                elif flag_0sim1 + flag_1sim2 + flag_2sim0 >= 2:
                    if frame_cnt not in output_dict:
                        output_dict[frame_cnt] = []
                    doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 3
                    dist_pred_fc = (dist_pred0[frame_cnt] + dist_pred1[frame_cnt] + dist_pred2[frame_cnt]) / 3
                    output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+self.nb_classes], doa_pred_fc[class_cnt+2*self.nb_classes], dist_pred_fc[class_cnt]])

        self.write_output_format_file(output_file, output_dict)
        
    def write_output_format_file(self, _output_format_file, _output_format_dict):
        """
        Writes DCASE output format csv file, given output format dictionary

        :param _output_format_file:
        :param _output_format_dict:
        :return:
        """
        _fid = open(_output_format_file, 'w')
        # _fid.write('{},{},{},{}\n'.format('frame number with 20ms hop (int)', 'class index (int)', 'azimuth angle (int)', 'elevation angle (int)'))
        for _frame_ind in _output_format_dict.keys():
            for _value in _output_format_dict[_frame_ind]:
                # Write Cartesian format output. Since baseline does not estimate track count and distance we use fixed values.
                _fid.write('{},{},{},{},{},{},{}\n'.format(int(_frame_ind), int(_value[0]), 0, float(_value[1]), float(_value[2]), float(_value[3]), float(_value[4])))
                # TODO: What if our system estimates track cound and distence (or only one of them)
        _fid.close()


class DataLoaderShuffleCallback(L.Callback):
    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        train_dataloader = trainer.train_dataloader
        shuffle_idx = np.arange(len(train_dataloader._circ_feat_queue))
        random.shuffle(shuffle_idx)
        train_dataloader._circ_feat_queue.shuffle(shuffle_idx)
        train_dataloader._circ_label_queue.shuffle(shuffle_idx)
        