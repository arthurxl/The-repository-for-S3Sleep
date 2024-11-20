import torch
from tqdm.auto import tqdm
from torch.optim import Optimizer
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, cohen_kappa_score, classification_report


class Trainer:
    def __init__(self, model, criterion, optimizer: Optimizer, mode, scheduler=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loss = []
        self.train_accuracy = []
        self.test_loss = []
        self.test_accuracy = []
        self.test_f1 = None
        self.test_per_class_f1 = None
        self.test_kappa = None
        self.test_cm = None
        self.test_per_class_acc = None
        self.mode = mode
        self.scheduler = scheduler
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _get_model_output(self, data, target):
        target = target.to(self.device)
        if self.mode == "patch" or self.mode == "class":
            output = self.model(data)

            loss = self.criterion(output, target)
        elif self.mode == "pyco":
            loss = 0
            y_temp = self.model(data)
            output = torch.zeros_like(y_temp[0])
            for j in range(len(y_temp)):
                loss += self.criterion(y_temp[j], target)
                output += y_temp[j]
        elif self.mode == 'frames':
            output = self.model(data).view(-1, 5)

            loss = self.criterion(output, target)
        elif self.mode == 'adaf':
            output, banlance_loss = self.model(data)
            output = output.reshape(-1, 5)
            loss = 0
            loss += self.criterion(output, target)
            loss += banlance_loss
        else:
            raise ValueError("Invalid mode{}".format(self.mode))
        return output, loss

    def train_one_epoch(self, dataloader):
        self.model.train()
        num = 0
        total_loss = 0
        total_accuracy = 0
        pbar = tqdm(enumerate(dataloader))
        for batch_idx, (data, target) in pbar:
            data = data.to(self.device)
            target = target.reshape(-1)
            num += len(target)
            # self.optimizer.zero_grad()
            output, loss = self._get_model_output(data, target)
            # loss.backward()
            if self.optimizer.__class__.__name__ == 'SAM':
                loss.backward()
                self.optimizer.first_step(zero_grad=True)

                output, loss = self._get_model_output(data, target)

                loss.backward()
                self.optimizer.second_step(zero_grad=True)
            else:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            total_loss += loss.item() * len(target)
            y_pred = output.argmax(dim=1, keepdim=False).detach().cpu()
            total_accuracy += y_pred.eq(target).sum().item()
            pbar.set_description(
                "batch:{}, acc:{:.4f}, loss:{:.4f}".format(batch_idx, total_accuracy / num, total_loss / num))
        if self.scheduler is not None:
            self.scheduler.step()
        self.train_loss.append(total_loss / num)
        self.train_accuracy.append(total_accuracy / num)

    def test_one_epoch(self, dataloader):
        self.model.eval()
        num = 0
        total_loss = 0
        total_accuracy = 0
        pbar = tqdm(enumerate(dataloader))
        y_true = []
        y_predicted = []
        for batch_idx, (data, target) in pbar:
            data = data.to(self.device)
            target = target.reshape(-1)
            with torch.no_grad():
                num += len(target)
                output, loss = self._get_model_output(data, target)
                total_loss += loss.item() * len(target)
                y_pred = output.argmax(dim=1, keepdim=False).detach().cpu()
                total_accuracy += y_pred.eq(target).sum().item()
                y_predicted += y_pred.cpu().tolist()
                y_true += target.cpu().tolist()

                pbar.set_description(
                    "batch:{}, acc:{:.4f}, loss{:.4f}".format(batch_idx, total_accuracy / num, total_loss / num))

        y_true = np.array(y_true, dtype=int)
        y_predicted = np.array(y_predicted, dtype=int)
        self.test_loss.append(total_loss / num)
        self.test_accuracy.append(total_accuracy / num)
        self.test_f1 = f1_score(y_true, y_predicted, average='macro')
        self.test_per_class_f1 = f1_score(y_true, y_predicted, average=None)
        self.test_kappa = cohen_kappa_score(y_true, y_predicted)
        self.test_cm = confusion_matrix(y_true, y_predicted)
        self.test_per_class_acc = classification_report(y_true, y_predicted, digits=4)
