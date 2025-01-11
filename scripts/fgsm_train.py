import argparse
import warnings
import sys
import os


def get_parser():
    parser = argparse.ArgumentParser(description="AIGCDetection @cby Training")
    parser.add_argument("--model_name", default='efficientnet-b0', help="Setting the model name", type=str)
    parser.add_argument("--embedding_size", default=None, help="Setting the embedding_size", type=int)
    parser.add_argument("--num_classes", default=2, help="Setting the num classes", type=int)
    parser.add_argument('--freeze_extractor', action='store_true', help='Whether to freeze extractor?')
    parser.add_argument("--model_path", default=None, help="Setting the model path", type=str)
    parser.add_argument('--no_strict', action='store_true', help='Whether to load model without strict?')
    parser.add_argument("--root_path", default='/disk4/chenby/dataset/MSCOCO',
                        help="Setting the root path for dataset loader", type=str)
    parser.add_argument("--fake_root_path", default='/disk4/chenby/dataset/DRCT-2M',
                        help="Setting the fake root path for dataset loader", type=str)
    parser.add_argument('--is_dire', action='store_true', help='Whether to using DIRE?')
    parser.add_argument("--regex", default='*.*', help="Setting the regex for dataset loader", type=str)
    parser.add_argument('--test_all', action='store_true', help='Whether to test_all?')
    parser.add_argument('--post_aug_mode', default=None, help='Stetting the post aug mode during test phase.')
    parser.add_argument('--save_txt', default=None, help='Stetting the save_txt path.')
    parser.add_argument("--fake_indexes", default='1',
                        help="Setting the fake indexes, multi class using '1,2,3,...' ", type=str)
    parser.add_argument("--dataset_name", default='MSCOCO', help="Setting the dataset name", type=str)
    parser.add_argument("--device_id", default='0',
                        help="Setting the GPU id, multi gpu split by ',', such as '0,1,2,3'", type=str)
    parser.add_argument("--input_size", default=224, help="Image input size", type=int)
    parser.add_argument('--is_crop', action='store_true', help='Whether to crop image?')
    parser.add_argument("--batch_size", default=64, help="Setting the batch size", type=int)
    parser.add_argument("--epoch_start", default=0, help="Setting the epoch start", type=int)
    parser.add_argument("--num_epochs", default=50, help="Setting the num epochs", type=int)
    parser.add_argument("--num_workers", default=4, help="Setting the num workers", type=int)
    parser.add_argument('--is_warmup', action='store_true', help='Whether to using lr warmup')
    parser.add_argument("--lr", default=1e-3, help="Setting the learning rate", type=float)
    parser.add_argument("--save_flag", default='', help="Setting the save flag", type=str)
    parser.add_argument("--sampler_mode", default='', help="Setting the sampler mode", type=str)
    parser.add_argument('--is_test', action='store_true', help='Whether to predict the test set?')
    parser.add_argument('--is_amp', action='store_true', help='Whether to using amp autocast(使用混合精度加速)?')
    parser.add_argument("--inpainting_dir", default='full_inpainting', help="rec_image dir", type=str)
    parser.add_argument("--threshold", default=0.5, help="Setting the valid or testing threshold.", type=float)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    return args


warnings.filterwarnings("ignore")
sys.path.append('..')
args = get_parser()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)

import torch
import torch.nn as nn
import torch.optim as optim
from catalyst.data import BalanceClassSampler
import time
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import gc
from sklearn.metrics import roc_auc_score, recall_score, precision_score, accuracy_score, f1_score
import pytorch_warmup as warmup
import matplotlib.pyplot as plt
import os

from utils.utils import Logger, AverageMeter, Test_time_agumentation, calculate_fnr
from network.models import get_models
from data.dataset import AIGCDetectionDataset, CLASS2LABEL_MAPPING, GenImage_LIST
from data.transform import create_train_transforms, create_val_transforms

def fgsm_attack(model, images, labels, eps=0.03):
    """
    使用FGSM生成对抗样本

    参数:
    - model: 要攻击的模型
    - images: 原始输入图像数据，形状为 (batch_size, channels, height, width)
    - labels: 对应的真实标签，形状为 (batch_size)
    - eps: 扰动的强度，即攻击的步长

    返回:
    - perturbed_images: 生成的对抗样本
    """
    images = images.clone().detach().requires_grad_(True)
    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    model.zero_grad()
    loss.backward()
    # 获取梯度的符号
    gradient_sign = images.grad.data.sign()
    # 生成对抗扰动并添加到原始图像上
    perturbed_images = images + eps * gradient_sign
    # 对图像进行裁剪，确保像素值在合法范围内（例如0到1之间，根据你的图像数据归一化情况而定）
    perturbed_images = torch.clamp(perturbed_images, 0, 1)
    return perturbed_images.detach()


def merge_tensor(img, label, is_train=True):
    def shuffle_tensor(img, label):
        indices = torch.randperm(img.size(0))
        return img[indices], label[indices]
    if isinstance(img, list) and isinstance(label, list):
        img, label = torch.cat(img, dim=0), torch.cat(label, dim=0)
        if is_train:
            img, label = shuffle_tensor(img, label)
    return img, label

# 9 times
def TTA(model_, img, activation=nn.Softmax(dim=1)):
    # original 1
    outputs = activation(model_(img))
    tta = Test_time_agumentation()
    # 水平翻转 + 垂直翻转 2
    flip_imgs = tta.tensor_flip(img)
    for flip_img in flip_imgs:
        outputs += activation(model_(flip_img))
    # 2*3=6
    for flip_img in [img, flip_imgs[0]]:
        rot_flip_imgs = tta.tensor_rotation(flip_img)
        for rot_flip_img in rot_flip_imgs:
            outputs += activation(model_(rot_flip_img))

    outputs /= 9

    return outputs


def eval_model_with_fgsm(model, epoch, eval_loader, is_save=True, is_tta=False, threshold=0.5, save_txt=None,
                         fgsm_eps=0.03):
    """
    在评估模型时，同时考虑原始数据和FGSM攻击生成的对抗样本进行评估

    参数:
    - model: 待评估的深度学习模型
    - epoch: 当前所处的训练轮次
    - eval_loader: 评估数据集的数据加载器
    - is_save: 是否保存评估结果到日志，默认为True
    - is_tta: 是否使用测试时增强技术，默认为False
    - threshold: 用于二分类判定的阈值，默认为0.5
    - save_txt: 若不为None，用于指定保存评估指标的文本文件路径，并返回相关指标；若为None，返回准确率
    - fgsm_eps: FGSM攻击的扰动强度（步长），默认为0.03

    返回:
    - 根据save_txt的值返回相应的评估指标或者准确率
    """
    model.eval()
    # 用于统计原始数据评估的损失值和准确率平均值
    losses_original = AverageMeter()
    accuracies_original = AverageMeter()
    # 用于统计FGSM对抗样本评估的损失值和准确率平均值
    losses_fgsm = AverageMeter()
    accuracies_fgsm = AverageMeter()
    eval_process = tqdm(eval_loader)
    labels_original = []
    outputs_original = []
    labels_fgsm = []
    outputs_fgsm = []

    with torch.no_grad():
        for i, (img, label) in enumerate(eval_process):
            img, label = merge_tensor(img, label, is_train=False)
            if i > 0 and i % 1 == 0:
                eval_process.set_description("Epoch: %d, Loss (Original): %.4f, Acc (Original): %.4f, "
                                             "Loss (FGSM): %.4f, Acc (FGSM): %.4f" %
                                             (epoch, losses_original.avg, accuracies_original.avg,
                                              losses_fgsm.avg, accuracies_fgsm.avg))
            img, label = img.cuda(), label.cuda()

            # 处理原始数据
            if not is_tta:
                y_pred_original = model(img)
                y_pred_original = nn.Softmax(dim=1)(y_pred_original)
            else:
                y_pred_original = TTA(model, img, activation=nn.Softmax(dim=1))
            outputs_original.append(1 - y_pred_original[:, 0])
            labels_original.append(label)
            loss_original = criterion(y_pred_original, label)
            acc_original = (torch.max(y_pred_original.detach(), 1)[1] == label).sum().item() / img.size(0)
            losses_original.update(loss_original.item(), img.size(0))
            accuracies_original.update(acc_original, img.size(0))

            # 生成并处理FGSM对抗样本
            with torch.set_grad_enabled(True):
                perturbed_img = fgsm_attack(model, img, label, eps=fgsm_eps)
            if not is_tta:
                y_pred_fgsm = model(perturbed_img)
                y_pred_fgsm = nn.Softmax(dim=1)(y_pred_fgsm)
            else:
                y_pred_fgsm = TTA(model, perturbed_img, activation=nn.Softmax(dim=1))
            outputs_fgsm.append(1 - y_pred_fgsm[:, 0])
            labels_fgsm.append(label)
            loss_fgsm = criterion(y_pred_fgsm, label)
            acc_fgsm = (torch.max(y_pred_fgsm.detach(), 1)[1] == label).sum().item() / perturbed_img.size(0)
            losses_fgsm.update(loss_fgsm.item(), perturbed_img.size(0))
            accuracies_fgsm.update(acc_fgsm, perturbed_img.size(0))


    # 处理原始数据的评估指标计算
    outputs_original = torch.cat(outputs_original, dim=0).cpu().numpy()
    labels_original = torch.cat(labels_original, dim=0).cpu().numpy()
    labels_original[labels_original > 0] = 1
    auc_original = roc_auc_score(labels_original, outputs_original)
    recall_original = recall_score(labels_original, outputs_original > threshold)
    precision_original = precision_score(labels_original, outputs_original > threshold)
    binary_acc_original = accuracy_score(labels_original, outputs_original > threshold)
    f1_original = f1_score(labels_original, outputs_original > threshold)
    fnr_original = calculate_fnr(labels_original, outputs_original > threshold)

    # 处理FGSM对抗样本的评估指标计算
    outputs_fgsm = torch.cat(outputs_fgsm, dim=0).cpu().numpy()
    labels_fgsm = torch.cat(labels_fgsm, dim=0).cpu().numpy()
    labels_fgsm[labels_fgsm > 0] = 1
    auc_fgsm = roc_auc_score(labels_fgsm, outputs_fgsm)
    recall_fgsm = recall_score(labels_fgsm, outputs_fgsm > threshold)
    precision_fgsm = precision_score(labels_fgsm, outputs_fgsm > threshold)
    binary_acc_fgsm = accuracy_score(labels_fgsm, outputs_fgsm > threshold)
    f1_fgsm = f1_score(labels_fgsm, outputs_fgsm > threshold)
    fnr_fgsm = calculate_fnr(labels_fgsm, outputs_fgsm > threshold)

    print(f'Original Data - AUC:{auc_original}-Recall:{recall_original}-Precision:{precision_original}-'
          f'BinaryAccuracy:{binary_acc_original}, f1: {f1_original}, fnr:{fnr_original}')
    print(f'FGSM Adversarial Examples - AUC:{auc_fgsm}-Recall:{recall_fgsm}-Precision:{precision_fgsm}-'
          f'BinaryAccuracy:{binary_acc_fgsm}, f1: {f1_fgsm}, fnr:{fnr_fgsm}')

    if is_save:
        train_logger.log(phase="val", values={
            'epoch': epoch,
            'loss_original': format(losses_original.avg, '.4f'),
            'acc_original': format(accuracies_original.avg, '.4f'),
            'loss_fgsm': format(losses_fgsm.avg, '.4f'),
            'acc_fgsm': format(accuracies_fgsm.avg, '.4f'),
            'lr': optimizer.param_groups[0]['lr']
        })
    print("Val (Original):\t Loss:{0:.4f} \t Acc:{1:.4f}".format(losses_original.avg, accuracies_original.avg))
    print("Val (FGSM):\t Loss:{0:.4f} \t Acc:{1:.4f}".format(losses_fgsm.avg, accuracies_fgsm.avg))

    acc_avg_original = accuracies_original.avg
    acc_avg_fgsm = accuracies_fgsm.avg

    # 垃圾回收
    del outputs_original, labels_original, losses_original, accuracies_original
    del outputs_fgsm, labels_fgsm, losses_fgsm, accuracies_fgsm
    gc.collect()

    if save_txt is not None:
        return binary_acc_original, auc_original, recall_original, precision_original, f1_original, fnr_original, \
               binary_acc_fgsm, auc_fgsm, recall_fgsm, precision_fgsm, f1_fgsm, fnr_fgsm
    return acc_avg_original, acc_avg_fgsm


def train_model(model, criterion, optimizer, epoch, scaler=None):
    model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    training_process = tqdm(train_loader)
    for i, (x, label) in enumerate(training_process):
        x, label = merge_tensor(x, label, is_train=True)
        optimizer.zero_grad()
        current_lr = optimizer.param_groups[0]['lr']
        if i > 0 and i % 1 == 0:
            training_process.set_description(
                "Epoch: %d, LR: %.8f, Loss: %.4f, Acc: %.4f" % (
                    epoch, current_lr, losses.avg, accuracies.avg))

        x = x.cuda()
        label = label.cuda()
        # label = Variable(torch.LongTensor(label).cuda(device_id))
        # Forward pass: Compute predicted y by passing x to the model
        if scaler is None:
            y_pred = model(x)
            # Compute and print loss
            loss = criterion(y_pred, label)
            acc = (torch.max(y_pred.detach(), 1)[1] == label).sum().item() / x.size(0)

            losses.update(loss.item(), x.size(0))  # loss -> loss.item(), 减少内存泄漏风险
            accuracies.update(acc, x.size(0))

            loss.backward()
            optimizer.step()
        else:
            with autocast():
                y_pred = model(x)
                # Compute and print loss
                loss = criterion(y_pred, label)
            acc = (torch.max(y_pred.detach(), 1)[1] == label).sum().item() / x.size(0)

            losses.update(loss.item(), x.size(0))
            accuracies.update(acc, x.size(0))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if args.is_warmup:
            with warmup_scheduler.dampening():
                scheduler.step()
    if not args.is_warmup:
        scheduler.step()
    train_logger.log(phase="train", values={
        'epoch': epoch,
        'loss': format(losses.avg, '.4f'),
        'acc': format(accuracies.avg, '.4f'),
        'lr': optimizer.param_groups[0]['lr']
    })
    print("Train:\t Loss:{0:.4f} \t Acc:{1:.4f}".format(losses.avg, accuracies.avg))
    # 垃圾回收
    del losses, accuracies
    gc.collect()


# python train.py --device_id=0 --model_name=efficientnet-b0 --input_size=224 --batch_size=48 --fake_indexes=1 --is_amp --save_flag=
if __name__ == '__main__':
    batch_size = args.batch_size * torch.cuda.device_count()
    writeFile = f"../output/{args.dataset_name}/{args.fake_indexes.replace(',', '_')}/" \
                f"{args.model_name.split('/')[-1]}_{args.input_size}{args.save_flag}/logs"
    store_name = writeFile.replace('/logs', '/weights')
    print(f'Using gpus:{args.device_id},batch size:{batch_size},gpu_count:{torch.cuda.device_count()},num_classes:{args.num_classes}')
    # Load model
    model = get_models(model_name=args.model_name, num_classes=args.num_classes,
                       freeze_extractor=args.freeze_extractor, embedding_size=args.embedding_size)
    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path, map_location='cpu'), strict=not args.no_strict)
        print('Model found in {}'.format(args.model_path))
    else:
        print('No model found, initializing random model.')
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    # criterion = LabelSmoothing(smoothing=0.05).cuda(device_id)
    is_train = not args.is_test
    if is_train:
        if store_name and not os.path.exists(store_name):
            os.makedirs(store_name)
        train_logger = Logger(model_name=writeFile, header=['epoch', 'loss', 'acc', 'lr'])
        # setting data loader
        xdl = AIGCDetectionDataset(args.root_path, fake_root_path=args.fake_root_path, fake_indexes=args.fake_indexes, phase='train',
                                   num_classes=args.num_classes, inpainting_dir=args.inpainting_dir, is_dire=args.is_dire,
                                   transform=create_train_transforms(size=args.input_size, is_crop=args.is_crop)
                                   )
        sampler = BalanceClassSampler(labels=xdl.get_labels(), mode=args.sampler_mode) if args.sampler_mode != '' else None  # "upsampling"
        train_loader = DataLoader(xdl, batch_size=batch_size, shuffle=sampler is None, num_workers=args.num_workers, sampler=sampler)
        train_dataset_len = len(xdl)

        xdl_eval = AIGCDetectionDataset(args.root_path, fake_root_path=args.fake_root_path, fake_indexes=args.fake_indexes, phase='val',
                                        num_classes=args.num_classes, inpainting_dir=args.inpainting_dir, is_dire=args.is_dire,
                                        transform=create_val_transforms(size=args.input_size, is_crop=args.is_crop)
                                        )
        eval_loader = DataLoader(xdl_eval, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
        eval_dataset_len = len(xdl_eval)
        print('train_dataset_len:', train_dataset_len, 'eval_dataset_len:', eval_dataset_len)

        # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=4e-5)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
        if not args.is_warmup:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5)
        else:
            num_steps = train_dataset_len * args.num_epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
            warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

        best_acc = 0.5 if args.epoch_start == 0 else eval_model(model, args.epoch_start - 1, eval_loader, is_save=False)
        for epoch in range(args.epoch_start, args.num_epochs):
            train_model(model, criterion, optimizer, epoch, scaler=GradScaler() if args.is_amp else None)
            if epoch % 1 == 0 or epoch == args.num_epochs - 1:
                acc = eval_model(model, epoch, eval_loader)
                if best_acc < acc:
                    best_acc = acc
                    save_path = '{}/{}_acc{:.4f}.pth'.format(store_name, epoch, acc)
                    if torch.cuda.device_count() > 1:
                        torch.save(model.module.state_dict(), save_path)
                    else:
                        torch.save(model.state_dict(), save_path)
            print(f'Current best acc:{best_acc}')
        last_save_path = '{}/last_acc{:.4f}.pth'.format(store_name, acc)
        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), last_save_path)
        else:
            torch.save(model.state_dict(), last_save_path)
    else:
        start = time.time()
        epoch_start = 1
        num_epochs = 1
        xdl_test = AIGCDetectionDataset(args.root_path, fake_root_path=args.fake_root_path, fake_indexes=args.fake_indexes,
                                        phase='test', num_classes=args.num_classes, is_dire=args.is_dire,
                                        post_aug_mode=args.post_aug_mode, regex=args.regex, inpainting_dir=args.inpainting_dir,
                                        transform=create_val_transforms(size=args.input_size, is_crop=args.is_crop)
                                        )
        test_loader = DataLoader(xdl_test, batch_size=batch_size, shuffle=False, num_workers=4)
        test_dataset_len = len(xdl_test)
        print('test_dataset_len:', test_dataset_len)
        out_metrics = eval_model_with_fgsm(model, epoch_start, test_loader, is_save=False, is_tta=False,
                                           threshold=args.threshold, save_txt=args.save_txt, fgsm_eps=0.03)
        print('Total time:', time.time() - start)
        # 保存测试结果，根据新的返回值情况进行相应调整
        if args.save_txt is not None:
            os.makedirs(os.path.dirname(args.save_txt), exist_ok=True)
            # 如果返回多个指标（原始数据和FGSM对抗样本的指标），进行相应的解包赋值
            if len(out_metrics) == 12:
                (binary_acc_original, auc_original, recall_original, precision_original, f1_original, fnr_original,
                 binary_acc_fgsm, auc_fgsm, recall_fgsm, precision_fgsm, f1_fgsm, fnr_fgsm) = out_metrics

                with open(args.save_txt, 'a') as file:
                    class_name = list(CLASS2LABEL_MAPPING.keys())[int(args.fake_indexes)]
                    result_str_original = f'model_path:{args.model_path}, post_aug_mode:{args.post_aug_mode}, class_name:{class_name}\n' \
                                          f'acc_original:{binary_acc_original:.4f}, auc_original:{auc_original:.4f}, recall_original:{recall_original:.4f}, precision_original:{precision_original:.4f}, ' \
                                          f'f1_original:{f1_original:.4f}, fnr_original: {fnr_original}\n'
                    result_str_fgsm = f'acc_fgsm:{binary_acc_fgsm:.4f}, auc_fgsm:{auc_fgsm:.4f}, recall_fgsm:{recall_fgsm:.4f}, precision_fgsm:{precision_fgsm:.4f}, ' \
                                      f'f1_fgsm:{f1_fgsm:.4f}, fnr_fgsm: {fnr_fgsm}\n'
                    file.write(result_str_original)
                    file.write(result_str_fgsm)
                print(f'The result was saved in {args.save_txt}')
            # 如果返回两个准确率（原始数据和FGSM对抗样本的准确率），进行相应处理
            elif len(out_metrics) == 2:
                acc_avg_original, acc_avg_fgsm = out_metrics
                with open(args.save_txt, 'a') as file:
                    class_name = list(CLASS2LABEL_MAPPING.keys())[int(args.fake_indexes)]
                    result_str_original = f'model_path:{args.model_path}, post_aug_mode:{args.post_aug_mode}, class_name:{class_name}\n' \
                                          f'acc_original:{acc_avg_original:.4f}\n'
                    result_str_fgsm = f'acc_fgsm:{acc_avg_fgsm:.4f}\n'
                    file.write(result_str_original)
                    file.write(result_str_fgsm)
                print(f'The result was saved in {args.save_txt}')

