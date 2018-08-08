import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import os
import os.path as osp

from tensorboardX import SummaryWriter
from datetime import datetime

from model.deeplab_multi import DeeplabMulti
from model.discriminator import FCDiscriminator
from utils.loss import CrossEntropy2d
from utils.common import mkdir_check
from dataset.list_dataset import ListDataSet

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'DeepLab'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 4
SRC_DATA_DIRECTORY = '/4TB/ytliou/crossdata_seg/dataset'
SRC_IMG_LIST_PATH = '/4TB/ytliou/crossdata_seg/list/cityscapes-train-img.txt'
SRC_LBL_LIST_PATH = '/4TB/ytliou/crossdata_seg/list/cityscapes-train-lbl.txt'
SRC_INPUT_SIZE = '1280,720' #'1024,512' 
IGNORE_LABEL = 255
TGT_DATA_DIRECTORY = '/4TB/ytliou/crossdata_seg/dataset'
TGT_IMG_LIST_PATH = '/4TB/ytliou/crossdata_seg/list/bdd-train-img.txt'
TGT_LBL_LIST_PATH = '/4TB/ytliou/crossdata_seg/list/bdd-train-lbl.txt'
TGT_IMG_NOLABEL_LIST_PATH = '/4TB/ytliou/crossdata_seg/list/bdd-train-img.txt'
TGT_INPUT_SIZE = '1280,720' #'1024,512'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 250000
NUM_STEPS_STOP = 80000  # early stopping
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 5000
SNAPSHOT_DIR = '/4TB/ytliou/snapshots/'
WEIGHT_DECAY = 0.0005

LEARNING_RATE_D = 1e-4
LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 0.0002
LAMBDA_ADV_TARGET2 = 0.001

TARGET = 'cityscapes'
SET = 'train'


def get_args():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--src-data-dir", type=str, default=SRC_DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--src-img-list", type=str, default=SRC_IMG_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--src-lbl-list", type=str, default=SRC_LBL_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--src-input-size", type=str, default=SRC_INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--tgt-data-dir", type=str, default=TGT_DATA_DIRECTORY,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--tgt-img-list", type=str, default=TGT_IMG_LIST_PATH,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--tgt-lbl-list", type=str, default=TGT_LBL_LIST_PATH,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--tgt-img-nolabel-list", type=str, default=TGT_IMG_NOLABEL_LIST_PATH,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--tgt-input-size", type=str, default=TGT_INPUT_SIZE,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
    parser.add_argument("--lambda-adv-target1", type=float, default=LAMBDA_ADV_TARGET1,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-adv-target2", type=float, default=LAMBDA_ADV_TARGET2,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    return parser.parse_args()

def loss_calc(pred, label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d().cuda(gpu)

    return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(args, optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(args, optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def main(args):
    """Create the model and start the training."""

    mkdir_check(args.snapshot_dir)
    writer = SummaryWriter(log_dir=os.path.join(args.snapshot_dir, 'tb-logs'))

    h, w = map(int, args.src_input_size.split(','))
    src_input_size = (h, w)

    h, w = map(int, args.tgt_input_size.split(','))
    tgt_input_size = (h, w)

    cudnn.enabled = True
    gpu = args.gpu

    # Create network
    if args.model == 'DeepLab':
        model = DeeplabMulti(num_classes=args.num_classes)
        if args.restore_from[:4] == 'http' :
            saved_state_dict = model_zoo.load_url(args.restore_from)
        else:
            saved_state_dict = torch.load(args.restore_from)

        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            # Scale.layer5.conv2d_list.3.weight
            i_parts = i.split('.')
            # print i_parts
            if not args.num_classes == 19 or not i_parts[1] == 'layer5':
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
                # print i_parts
        model.load_state_dict(new_params)

    model.train()
    model.cuda(args.gpu)

    cudnn.benchmark = True

    # init D
    model_D2 = FCDiscriminator(num_classes=19)

    model_D2.train()
    model_D2.cuda(args.gpu)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    trainloader = data.DataLoader(
            ListDataSet(args.src_data_dir, args.src_img_list, args.src_lbl_list, 
                max_iters=args.num_steps * args.iter_size * args.batch_size, crop_size=src_input_size, mean=IMG_MEAN),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    trainloader_iter = enumerate(trainloader)

    targetloader = data.DataLoader(
            ListDataSet(args.tgt_data_dir, args.tgt_img_list, args.tgt_lbl_list,
                max_iters=args.num_steps * args.iter_size * args.batch_size * 2, crop_size=tgt_input_size, mean=IMG_MEAN),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
            pin_memory=True)

    targetloader_nolabel = data.DataLoader(
            ListDataSet(args.tgt_data_dir, args.tgt_img_nolabel_list, None,
                max_iters=args.num_steps * args.iter_size * args.batch_size * 2, crop_size=tgt_input_size, mean=IMG_MEAN),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
            pin_memory=True)

    targetloader_iter = enumerate(targetloader)
    targetloader_nolabel_iter = enumerate(targetloader_nolabel)

    # implement model.optim_parameters(args) to handle different models' lr setting

    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    optimizer_D2 = optim.Adam(model_D2.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D2.zero_grad()

    bce_loss = torch.nn.BCEWithLogitsLoss()

    interp = nn.Upsample(size=(src_input_size[1], src_input_size[0]), mode='bilinear')
    interp_target = nn.Upsample(size=(tgt_input_size[1], tgt_input_size[0]), mode='bilinear')

    # labels for adversarial training
    source_label = 0
    target_label = 1

    for i_iter in range(args.num_steps):

        loss_seg_value2 = 0
        loss_tgt_seg_value2 = 0
        loss_adv_target_value2 = 0
        loss_D_value2 = 0

        optimizer.zero_grad()
        adjust_learning_rate(args, optimizer, i_iter)

        optimizer_D2.zero_grad()
        adjust_learning_rate_D(args, optimizer_D2, i_iter)

        for sub_i in range(args.iter_size):

            # train G

            # don't accumulate grads in D
            for param in model_D2.parameters():
                param.requires_grad = False

            # train with source

            _, batch = trainloader_iter.__next__()
            images, labels, _, _ = batch
            images = Variable(images).cuda(args.gpu)

            pred1, pred2 = model(images)
            #pred1 = interp(pred1)
            pred2 = interp(pred2)

            #loss_seg1 = loss_calc(pred1, labels, args.gpu)
            loss_seg2 = loss_calc(pred2, labels, args.gpu)
            loss = loss_seg2 #+ args.lambda_seg * loss_seg1

            # proper normalization
            loss = loss / args.iter_size
            loss.backward()
            loss_seg_value2 += loss_seg2.data.cpu().numpy()[0] / args.iter_size

            # train with target seg

            _, batch = targetloader_iter.__next__()
            images, labels, _, _ = batch
            images = Variable(images).cuda(args.gpu)

            pred_target1, pred_target2 = model(images)
            #pred_target1 = interp_target(pred_target1)
            pred_target2 = interp_target(pred_target2)

            #loss_tgt_seg1 = loss_calc(pred_target1, labels, args.gpu)
            loss_tgt_seg2 = loss_calc(pred_target2, labels, args.gpu)
            loss = loss_tgt_seg2 #+ args.lambda_seg * loss_tgt_seg1

            # proper normalization
            loss = loss / args.iter_size
            loss.backward(retain_graph=True)
            loss_tgt_seg_value2 += loss_tgt_seg2.data.cpu().numpy()[0] / args.iter_size

            # train with target_nolabel adv
            _, batch = targetloader_nolabel_iter.__next__()
            images, _, _, _ = batch
            images = Variable(images).cuda(args.gpu)

            pred_target1, pred_target2 = model(images)
            #pred_target1 = interp_target(pred_target1)
            pred_target2 = interp_target(pred_target2)

            D_out2 = model_D2(F.softmax(pred_target2))

            loss_adv_target2 = bce_loss(D_out2,
                                        Variable(torch.FloatTensor(D_out2.data.size()).fill_(source_label)).cuda(
                                            args.gpu))

            loss = args.lambda_adv_target2 * loss_adv_target2
            loss = loss / args.iter_size
            loss.backward()
            loss_adv_target_value2 += loss_adv_target2.data.cpu().numpy()[0] / args.iter_size

            # train D

            # bring back requires_grad
            for param in model_D2.parameters():
                param.requires_grad = True

            # train with source
            pred2 = pred2.detach()

            D_out2 = model_D2(F.softmax(pred2))

            loss_D2 = bce_loss(D_out2,
                               Variable(torch.FloatTensor(D_out2.data.size()).fill_(source_label)).cuda(args.gpu))

            loss_D2 = loss_D2 / args.iter_size / 2

            loss_D2.backward()

            loss_D_value2 += loss_D2.data.cpu().numpy()[0]

            # train with target
            pred_target2 = pred_target2.detach()

            D_out2 = model_D2(F.softmax(pred_target2))

            loss_D2 = bce_loss(D_out2,
                               Variable(torch.FloatTensor(D_out2.data.size()).fill_(target_label)).cuda(args.gpu))

            loss_D2 = loss_D2 / args.iter_size / 2

            loss_D2.backward()

            loss_D_value2 += loss_D2.data.cpu().numpy()[0]

        optimizer.step()
        optimizer_D2.step()

        print('exp = {}'.format(args.snapshot_dir))
        print(
        'iter = {:5d}/{:8d}, loss_seg2 = {:.3f} loss_tgt_seg2 = {:.3f} loss_adv2 = {:.3f} loss_D2 = {:.3f}'.format(
            i_iter, args.num_steps_stop, loss_seg_value2, loss_tgt_seg_value2,
            loss_adv_target_value2, loss_D_value2))

        writer.add_scalars('loss/seg', {
            'src2': loss_seg_value2, 
            'tgt2': loss_tgt_seg_value2,
            }, i_iter)

        writer.add_scalar('loss/adv', loss_adv_target_value2, i_iter)
        writer.add_scalar('loss/d', loss_D_value2, i_iter)

        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'model_{}.pth'.format(args.num_steps)))
            torch.save(model_D2.state_dict(), osp.join(args.snapshot_dir, 'model_d_{}.pth'.format(args.num_steps)))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'model_{}.pth'.format(args.num_steps)))
            torch.save(model_D2.state_dict(), osp.join(args.snapshot_dir, 'model_d_{}.pth'.format(args.num_steps)))

if __name__ == '__main__':
    main(get_args())
