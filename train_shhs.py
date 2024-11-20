import torch

import train

import glob
import json
import time

import torch
from sklearn.model_selection import KFold, train_test_split

from CustomData import CustomDataset
from TranCon import RegisterTransformer
from Pyramid import MainModel
from file_load import load_npz_list_files, load_edf_file
from tools import *
from trainer import Trainer
from CPC_model import CPC_Cls
from AttnSleep import SeqAttn
import gc

if __name__ == '__main__':
    # with open(r'configs\config_pyco.json') as config_file:
    #     config = json.load(config_file)
    setup_seed(3407)
    args = train.parse_args()
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
    train_data_list, test_data_list = train_test_split(data_list, test_size=100)
    train_data_ = load_npz_list_files(npz_files=train_data_list, EEG='x', select_label='y')
    test_data_ = load_npz_list_files(npz_files=test_data_list, EEG='x', select_label='y')
    # kf = KFold(n_splits=args.k_folds)
    kf_fold = 0
    result = []
    best_list = []
    test_kappa_list = []
    # for train_index, test_index in kf.split(dataset):
    #     kf_fold = kf_fold + 1
    # dataset = CustomDataset(data[0], data[1], args.seq_len)
    train_data = CustomDataset(train_data_[0], train_data_[1], args.seq_len)
    test_data = CustomDataset(test_data_[0], test_data_[1], args.seq_len)
    args.cur_out_path = os.path.join(args.out_path, "shhs")
    args.best_path = os.path.join(args.cur_out_path, "best")
    os.makedirs(args.cur_out_path, exist_ok=True)
    model = SeqAttn().to(args.device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate,
                                 weight_decay=1e-3, eps=1e-8, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.1 if epoch >= 10 else 1)
    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, mode="frames", scheduler=scheduler)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
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
            save_cur_model(args, model, best=True)
        if epoch == args.num_epochs - 1:
            result.append(trainer.test_accuracy[-1])
            # save_cur_model(args=args, model=model, best=False)
    polt_data(data=trainer.train_accuracy, save_path=os.path.join(args.cur_out_path, "train"), is_save=True)
    polt_data(data=trainer.test_accuracy, save_path=os.path.join(args.cur_out_path, "test"), is_save=True)
    polt_data(data=trainer.train_loss, save_path=os.path.join(args.cur_out_path, "train_loss"), is_save=True)
    polt_data(data=trainer.test_loss, save_path=os.path.join(args.cur_out_path, "test_loss"), is_save=True)
    print("result:{}".format(trainer.test_accuracy[-1]))
    print("best:{}".format(best))
    with open(args.log_path, "a") as f:
        f.write("result:{}\n".format(trainer.test_accuracy[-1]))
        f.write("best:{}\n".format(best))
        f.write(f"kappa:{best_kappa}\n")
        f.write(f"cm:{best_cm}\n")
