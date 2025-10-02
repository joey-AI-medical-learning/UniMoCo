# coding=utf-8
import argparse
import os
import time
import logging
import random
import numpy as np
import pandas as pd
import torch
import torch.optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import model_class
from dataset.transforms import *
from dataset.dataset_nii import BraTS_load_all_train_nii_class, Brats_load_all_test_nii_class
from dataset.data_utils import init_fn
from utils import Parser, criterions
from utils.parser import setup
from utils.lr_scheduler import LR_Scheduler, record_loss, MultiEpochsDataLoader
from predict_class import AverageMeter, test_dice_hd95_softmax


parser = argparse.ArgumentParser()

parser.add_argument('-batch_size', '--batch_size', default=1, type=int, help='Batch size')
parser.add_argument('--datapath', default='/data', type=str)
parser.add_argument('--dataname', default='', type=str)
parser.add_argument('--savepath', default='', type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--pretrain', default=None, type=str)
parser.add_argument('--model', default='UniMoCo', type=str)
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--iter_per_epoch', default=50, type=int)
parser.add_argument('--region_fusion_start_epoch', default=0, type=int)
parser.add_argument('--seed', default=1024, type=int)
path = os.path.dirname(__file__)

args = parser.parse_args()
setup(args, 'training')
args.train_transforms = 'Compose([RandCrop3D((128,128,128)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ckpts = args.savepath
os.makedirs(ckpts, exist_ok=True)

writer = SummaryWriter(os.path.join(args.savepath, 'summary'))

masks = [[False, False, False, True], [False, True, False, False], [False, False, True, False],
         [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True],
         [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]
masks_torch = torch.from_numpy(np.array(masks))
mask_name = ['t2', 't1c', 't1', 'flair',
             't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
             'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
             'flairt1cet1t2']
print(masks_torch.int())


def main():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    if args.dataname in ['BRATS2021', 'BRATS2020', 'BRATS2018','ucsf_pdgm']:
        num_cls = 4
    else:
        print('dataset is error')
        exit(0)
    model = model_class.Model(num_cls=num_cls)
    print(model)
    model = torch.nn.DataParallel(model, device_ids=[0])

    lr_schedule = LR_Scheduler(args.lr, args.num_epochs)
    train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay}]
    optimizer = torch.optim.Adam(train_params, betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
    class_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    if args.dataname in ['BRATS2020', 'BRATS2015', 'ucsf_pdgm']:
        train_file = 'IDH_Train.xlsx'
        test_file = 'IDH_Test.xlsx'

    logging.info(str(args))

    train_set = BraTS_load_all_train_nii_class(transforms=args.train_transforms, root=args.datapath, num_cls=num_cls,
                                         train_file=train_file)
    test_set = Brats_load_all_test_nii_class(transforms=args.test_transforms, root=args.datapath, test_file=test_file)

    train_loader = MultiEpochsDataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=init_fn)
    test_loader = MultiEpochsDataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        logging.info('best epoch: {}'.format(checkpoint['epoch']))
        model.load_state_dict(checkpoint['state_dict'])

    start = time.time()
    torch.set_grad_enabled(True)
    logging.info('#############training############')
    iter_per_epoch = len(train_loader)
    train_iter = iter(train_loader)
    for epoch in range(args.num_epochs):
        step_lr = lr_schedule(optimizer, epoch)
        writer.add_scalar('lr', step_lr, global_step=(epoch + 1))
        b = time.time()
        for i in range(iter_per_epoch):
            step = (i + 1) + epoch * iter_per_epoch
            try:
                data = next(train_iter)
            except:
                train_iter = iter(train_loader)
                data = next(train_iter)
            x, target, class_label, mask = data[:4]
            x = x.to(device)
            target = target.to(device)
            class_label = class_label.to(device)
            mask = mask.to(device)

            model.module.is_training = True

            fuse_pred, sep_preds, prm_preds, spe_out_kl, kl_pred_fuse, fuse_class_output = model(x, mask)

            fuse_cross_loss = criterions.softmax_weighted_loss(fuse_pred, target, num_cls=num_cls)
            fuse_dice_loss = criterions.dice_loss(fuse_pred, target, num_cls=num_cls)
            fuse_loss = fuse_cross_loss + fuse_dice_loss

            sep_cross_loss = torch.zeros(1).to(device).float()
            sep_dice_loss = torch.zeros(1).to(device).float()
            for sep_pred in sep_preds:
                sep_cross_loss += criterions.softmax_weighted_loss(sep_pred, target, num_cls=num_cls)
                sep_dice_loss += criterions.dice_loss(sep_pred, target, num_cls=num_cls)
            sep_loss = sep_cross_loss + sep_dice_loss

            prm_cross_loss = torch.zeros(1).to(device).float()
            prm_dice_loss = torch.zeros(1).to(device).float()
            for prm_pred in prm_preds:
                prm_cross_loss += criterions.softmax_weighted_loss(prm_pred, target, num_cls=num_cls).to(device)
                prm_dice_loss += criterions.dice_loss(prm_pred, target, num_cls=num_cls).to(device)
            prm_loss = prm_cross_loss + prm_dice_loss

            mse_loss = torch.zeros(1).to(device).float()
            for sep_pre in sep_preds:
                mse_loss += F.mse_loss(sep_pre, fuse_pred).to(device)

            kl_loss = torch.zeros(1).to(device).float()
            weight_prm = 1.0
            for kl_pred, kl_spe in zip(kl_pred_fuse, spe_out_kl):
                weight_prm /= 2.0
                kl_loss += weight_prm * criterions.temp_kl_loss_bs(kl_pred, kl_spe).to(device)

            fuse_class_cross_loss = class_loss(fuse_class_output, class_label)

            if epoch < args.region_fusion_start_epoch:
                loss = fuse_loss * 0.0 + sep_loss + prm_loss + mse_loss + kl_loss + fuse_class_cross_loss
            else:
                loss = fuse_loss + sep_loss + prm_loss + mse_loss + kl_loss + fuse_class_cross_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('loss', loss.item(), global_step=step)
            writer.add_scalar('fuse_cross_loss', fuse_cross_loss.item(), global_step=step)
            writer.add_scalar('fuse_dice_loss', fuse_dice_loss.item(), global_step=step)
            writer.add_scalar('sep_cross_loss', sep_cross_loss.item(), global_step=step)
            writer.add_scalar('sep_dice_loss', sep_dice_loss.item(), global_step=step)
            writer.add_scalar('prm_cross_loss', prm_cross_loss.item(), global_step=step)
            writer.add_scalar('prm_dice_loss', prm_dice_loss.item(), global_step=step)
            writer.add_scalar('mse_loss', mse_loss.item(), global_step=step)
            writer.add_scalar('kl_loss', kl_loss.item(), global_step=step)
            writer.add_scalar('fuse_class_cross_loss', fuse_class_cross_loss.item(), global_step=step)


            msg = 'Epoch {}/{}, Iter {}/{}, Loss {:.4f}, '.format((epoch + 1), args.num_epochs, (i + 1), iter_per_epoch,
                                                                  loss.item())
            msg += 'fusecross:{:.4f}, fusedice:{:.4f}, '.format(fuse_cross_loss.item(), fuse_dice_loss.item())
            msg += 'sepcross:{:.4f}, sepdice:{:.4f}, '.format(sep_cross_loss.item(), sep_dice_loss.item())
            msg += 'prmcross:{:.4f}, prmdice:{:.4f}, '.format(prm_cross_loss.item(), prm_dice_loss.item())
            msg += 'mseloss:{:.4f}, '.format(mse_loss.item())
            msg += 'klloss:{:.4f}, '.format(kl_loss.item())

            msg += 'fuse_class_cross_loss:{:.4f}, '.format(
                fuse_class_cross_loss.item())

            logging.info(msg)
        logging.info('train time per epoch: {}'.format(time.time() - b))

        file_name = os.path.join(ckpts, 'model_last.pth')
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
        },
            file_name)

        if (epoch + 1) % 100 == 0 or (epoch >= (args.num_epochs - 5)):
            file_name = os.path.join(ckpts, 'model_{}.pth'.format(epoch + 1))
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
            },
                file_name)

    msg = 'total time: {:.4f} hours'.format((time.time() - start) / 3600)
    logging.info(msg)

    #  Evaluate the last epoch model
    test_dice_score = AverageMeter()
    test_accuracy_score = AverageMeter()
    test_precision_score = AverageMeter()
    test_recall_score = AverageMeter()
    test_f1_score = AverageMeter()
    csv_name = os.path.join(ckpts, '{}.csv'.format(args.model))
    with torch.no_grad():
        logging.info('###########test set wi/wo postprocess###########')
        for i, mask in enumerate(masks[::-1]):
            logging.info('{}'.format(mask_name[::-1][i]))
            dice_score, accuracy_score, precision_score, recall_score, f1_score = test_dice_hd95_softmax(
                test_loader,
                model,
                dataname=args.dataname,
                feature_mask=mask,
                mask_name=mask_name[::-1][i],
                csv_name=csv_name
            )
            test_dice_score.update(dice_score)
            test_accuracy_score.update(accuracy_score)
            test_precision_score.update(precision_score)
            test_recall_score.update(recall_score)
            test_f1_score.update(f1_score)

        logging.info('Avg Dice scores: {}'.format(test_dice_score.avg))
        logging.info('Avg Accuracy scores: {}'.format(test_accuracy_score.avg))
        logging.info('Avg Precision scores: {}'.format(test_precision_score.avg))
        logging.info('Avg Recall scores: {}'.format(test_recall_score.avg))
        logging.info('Avg F1 scores: {}'.format(test_f1_score.avg))


if __name__ == '__main__':
    main()
