import argparse
import glob
import json
import os.path
import time

import numpy as np
import torch
from sklearn.model_selection import KFold
from sam import SAM
from CustomData import CustomDataset
from file_load import load_npz_list_files, load_edf_file
from tools import setup_seed, polt_data, save_cur_model
from trainer import Trainer

from safetensors.torch import load_file
from s3 import *
import gc

CUDA_LAUNCH_BLOCKING = 1


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--negative-samples', type=int, default=15)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--seq-len', type=int, default=1)
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--k-folds', type=int, default=20)
    parser.add_argument('--device', type=int, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--out', type=str, default='')
    parser.add_argument('--data_path', type=str,
                        default=r'E:\dataset\sleepedf20\*.npz')
    parser.add_argument('--patch_size', type=int, default=10)
    parser.add_argument('--labels_path', type=str,
                        default=r'E:\20240313_guoxiaoyu_final\data\sleepEDF-78\data_array\raw_data\labels\*.npy')
    parser.add_argument('--pretrain_path', type=str, default=r'E:\Project\contrastive\logs\2024-06-03\10-23-04')
    args_parsed = parser.parse_args()
    return args_parsed


if __name__ == '__main__':
    with open(r'configs\config_pyco.json') as config_file:
        config = json.load(config_file)
    setup_seed(3407)
    args = parse_args()
    args.time = time.strftime("%Y-%m-%d", time.localtime())
    args.time = os.path.join("logs", args.time)

    args.cur_time = time.strftime("%H-%M-%S", time.localtime())
    args.out_path = os.path.join(args.time, args.cur_time)
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    args.log_path = os.path.join(args.out_path, "log.txt")
    with open(args.log_path, "a") as f:
        f.write("num epochs:{}\n".format(args.num_epochs))
        f.write("lr:{}\n".format(args.learning_rate))
        f.write("batch size:{}\n".format(args.batch_size))
        f.write("dataset:{}\n".format(args.data_path))
    data_list = glob.glob(args.data_path)
    data = load_npz_list_files(npz_files=data_list, EEG='x', select_label='y')

    # dataset = CustomDataset(data[0], data[1], args.seq_len)
    # 人的训练
    # label_list = glob.glob(args.labels_path)
    # data_person = load_edf_file(data_list, label_list)
    # dataset = CustomDataset(data_person[0], data_person[1], args.seq_len)

    kf = KFold(n_splits=args.k_folds)
    kf_fold = 0
    result = []
    best_list = []
    test_kappa_list = []
    # for train_index, test_index in kf.split(dataset):
    #     kf_fold = kf_fold + 1
    cm = np.zeros((5, 5), dtype=np.int)
    for i in range(args.k_folds):
        kf_fold = kf_fold + 1

        # if i!=11:
        #     continue
        for i_fold, (train_index, test_index) in enumerate(kf.split(data_list)):
            if i_fold == i:
                all_test_npz_path = []
                for k in range(0, len(test_index)):
                    all_test_npz_path.append(data_list[test_index[k]])

                all_train_npz_path = np.setdiff1d(data_list, all_test_npz_path)
        data_train, label_train = load_npz_list_files(all_train_npz_path, 'x', 'y')
        data_test, label_test = load_npz_list_files(all_test_npz_path, 'x', 'y')
        train_data = CustomDataset(data_train, label_train, args.seq_len)
        test_data = CustomDataset(data_test, label_test, args.seq_len)
        # args.cur_pretrain_path = os.path.join(args.pretrain_path, '{}/checkpoint_20.safetensors'.format(kf_fold))
        # state_dict = load_file(args.cur_pretrain_path)
        # filtered_state_dict = {k: v for k, v in state_dict.items() if 'classifier' not in k}
        # train_data = torch.utils.data.Subset(dataset, train_index)
        # test_data = torch.utils.data.Subset(dataset, test_index)
        # model = SuperviseTransformer(patch_size=args.patch_size, seq_len=3000, heads=8, num_classes=5,
        #                              dim_feedforward=512,dropout=0.3).to(args.device)
        # model = CPC_Cls().to(args.device)
        # model = EpochNet().to(args.device)
        args.cur_out_path = os.path.join(args.out_path, str(kf_fold))
        args.best_path = os.path.join(args.cur_out_path, "best")
        os.makedirs(args.cur_out_path, exist_ok=True)
        # model = MainModel(config).to(args.device)
        model = S3().to(args.device)
        # model = AttnSleep_pa().to(args.device)
        # model = SeqAttn().to(args.device)
        # model = AttnSleepAda().to(args.device)
        # model.load_state_dict(filtered_state_dict, strict=False)
        # model = ClassBackbone(config).to(args.device)
        # model = RegisterTransformer(patch_size=10, depth=8, heads=8).to(args.device)
        # param = [{'params': model.feature.parameters(), 'lr': 1e-4},
        #          {'params': model.classifier.parameters(), 'lr': 5e-4}]
        # feature_list = ["conv_c5.weight", "conv_c5.bias", "conv_c4.weight", "conv_c4.bias", "conv_c3.weight",
        #                 "conv_c3.bias"]
        # for name, p in model.feature.named_parameters():
        #     if name not in feature_list:
        #         p.requires_grad = False
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate,
                                     weight_decay=1e-3, eps=1e-8, amsgrad=True)
        # optimizer = SAM(params=model.parameters(), base_optimizer=torch.optim.Adam, rho=0.05, lr=args.learning_rate,
        #                 weight_decay=1e-3, eps=1e-8)
        # weight = torch.FloatTensor([1, 1, 6, 1, 1]).to("cuda")
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.1 if epoch >= 10 else 1)
        trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, mode="frames",scheduler=scheduler)
        train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
        del data_train, label_train, data_test, label_test, train_data, test_data
        gc.collect()

        best = 0
        best_cm = None
        best_kappa = 0
        for epoch in range(0, args.num_epochs):
            args.cur_epoch = epoch + 1
            print('Epoch {}/{} k_fold{}'.format(epoch + 1, args.num_epochs, kf_fold))
            trainer.train_one_epoch(train_data_loader)
            trainer.test_one_epoch(test_data_loader)
            if best < trainer.test_accuracy[-1]:
                best = trainer.test_accuracy[-1]
                best_cm = trainer.test_cm
                best_kappa = trainer.test_kappa
                # save_cur_model(args, model, best=True)
            if epoch == args.num_epochs - 1:
                result.append(trainer.test_accuracy[-1])
        best_list.append(best)
        test_kappa_list.append(best_kappa)
        cm += best_cm

        # save_cur_model(args=args, model=model, best=False)
        polt_data(data=trainer.train_accuracy, save_path=os.path.join(args.cur_out_path, "train"), is_save=True)
        polt_data(data=trainer.test_accuracy, save_path=os.path.join(args.cur_out_path, "test"), is_save=True)
        polt_data(data=trainer.train_loss, save_path=os.path.join(args.cur_out_path, "train_loss"), is_save=True)
        polt_data(data=trainer.test_loss, save_path=os.path.join(args.cur_out_path, "test_loss"), is_save=True)
    result = np.array(result)

    print("result:{}".format(result.sum() / kf_fold))
    print("best:{}".format(sum(best_list) / len(best_list)))
    with open(args.log_path, "a") as f:
        f.write("result:{}\n".format(result.sum() / kf_fold))
        f.write("best:{}\n".format(sum(best_list) / len(best_list)))
        f.write(f"kappa:{sum(test_kappa_list)/len(test_kappa_list)}\n")

        f.write(f"cm:{cm}\n")
