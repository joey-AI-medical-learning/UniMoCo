import torch
import logging
from predict_segmentation import AverageMeter, test_dice_hd95_softmax
from dataset.dataset_nii import BraTS_load_all_test_nii_segmentation
from utils.lr_scheduler import LR_Scheduler, record_loss, MultiEpochsDataLoader
from model import model_segmentation
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    masks = [[False, False, False, True], [False, True, False, False], [False, False, True, False],
             [True, False, False, False],
             [False, True, False, True], [False, True, True, False], [True, False, True, False],
             [False, False, True, True], [True, False, False, True], [True, True, False, False],
             [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
             [True, True, True, True]]
    mask_name = ['t2', 't1c', 't1', 'flair',
                 't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
                 'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
                 'flairt1cet1t2']

    test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'
    datapath = '/data'
    test_file = ''
    resume = 'model_last.pth'
    num_cls = 4
    dataname = 'ucsf_pdgm'
    nii_file = ''
    os.makedirs(nii_file, exist_ok=True)

    test_set = BraTS_load_all_test_nii_segmentation(transforms=test_transforms, root=datapath, test_file=test_file)
    test_loader = MultiEpochsDataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    model = model_segmentation.Model(num_cls=num_cls)
    model = torch.nn.DataParallel(model, device_ids=[0])
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['state_dict'])

    test_dice_score = AverageMeter()
    test_hd95_score = AverageMeter()
    csv_name = 'segmentation.csv'
    with torch.no_grad():
        logging.info('#################test last epoch model######################')
        for i, mask in enumerate(masks[::-1]):
            print('{}'.format(mask_name[::-1][i]))
            dice_score = test_dice_hd95_softmax(
                test_loader,
                model,
                output_file=nii_file,
                dataname=dataname,
                feature_mask=mask,
                mask_name=mask_name[::-1][i],
                csv_name=csv_name
            )
            test_dice_score.update(dice_score)

        logging.info('Avg Dice scores: {}'.format(test_dice_score.avg))
