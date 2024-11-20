import random
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from safetensors.torch import save_model


def zigzag_path(M, N):
    def zigzag_path_lr(M, N, start_row=0, start_col=0, dir_row=1, dir_col=1):
        path = []
        for i in range(N):
            for j in range(M):
                # If the row number is even, move right; otherwise, move left
                col = j if i % 2 == 0 else M - 1 - j
                path.append((start_row + dir_row * i) * M + start_col + dir_col * col)
        return path

    def zigzag_path_tb(M, N, start_row=0, start_col=0, dir_row=1, dir_col=1):
        path = []
        for j in range(M):
            for i in range(N):
                # If the column number is even, move down; otherwise, move up
                row = i if j % 2 == 0 else N - 1 - i
                path.append((start_row + dir_row * row) * M + start_col + dir_col * j)
        return path

    paths = []
    for start_row, start_col, dir_row, dir_col in [
        (0, 0, 1, 1),
        (0, M - 1, 1, -1),
        (N - 1, 0, -1, 1),
        (N - 1, M - 1, -1, -1),
    ]:
        paths.append(zigzag_path_lr(M, N, start_row, start_col, dir_row, dir_col))
        paths.append(zigzag_path_tb(M, N, start_row, start_col, dir_row, dir_col))

    for _index, _p in enumerate(paths):
        paths[_index] = np.array(_p)

    _zz_paths = paths[:8]

    return _zz_paths


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def covert_label(labels):
    labels_int = []
    for c in labels:
        if c == 'w':
            c = int(0)
        elif c == '1':
            c = int(0)
        elif c == 'n':
            c = int(1)
        elif c == '2':
            c = int(1)
        elif c == 'r':
            c = int(2)
        elif c == '3':
            c = int(2)
        #
        # elif c=='0':
        #     c = int(0)
        # elif c=='2':
        #     c = int(2)
        else:
            c = int(2)  # 伪迹 Artifact
            # c = int(4)  #伪迹 Artifact
        labels_int.append(c)
    labels_np = np.array(labels_int).astype("int8")
    return labels_np


def save_cur_model(args, model, best=False):
    if not os.path.exists(args.cur_out_path):
        os.makedirs(args.cur_out_path)
    if best:
        out = os.path.join(args.best_path, "best_checkpoint.safetensors")
        os.makedirs(args.best_path, exist_ok=True)
        with open(os.path.join(args.best_path, "best_checkpoint.txt"), "a") as f:
            f.write("{}\n".format(str(args.cur_epoch)))
    else:
        out = os.path.join(args.cur_out_path, "checkpoint_{}.safetensors".format(args.cur_epoch))

    save_model(model, out)


def polt_data(data, save_path: str = None, is_save=False):
    plt.figure(figsize=(10, 6))
    plt.plot(data)

    plt.title(os.path.basename(save_path))
    if is_save:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    polt_data(data)
