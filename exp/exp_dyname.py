import os
import time
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.data_loader import Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute
from exp.exp_basic import Exp_Basic
from models.dyname import DynaME
from utils.metrics import cumavg, metric
from utils.tools import EarlyStopping, adjust_learning_rate


class Exp_TS2VecSupervised(Exp_Basic):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.online = args.online_learning
        self.model = DynaME(args, self.device).to(self.device)
        self.past = None
        self.opt = None
        self.opt_gate = None

    def _get_data(self, flag):
        mapping = defaultdict(lambda: Dataset_Custom, {
            "ETTh1": Dataset_ETT_hour, "ETTh2": Dataset_ETT_hour,
            "ETTm1": Dataset_ETT_minute, "ETTm2": Dataset_ETT_minute,
        })
        dataset = mapping[self.args.data](
            root_path=self.args.root_path, data_path=self.args.data_path, flag=flag,
            delay_fb=self.args.delay_fb,
            size=[self.args.seq_len, self.args.label_len, self.args.pred_len],
            features=self.args.features, target=self.args.target, inverse=self.args.inverse,
            timeenc=2, freq=self.args.freq, cols=self.args.cols,
        )
        batch_size = self.args.test_bsz if flag == "test" else self.args.batch_size
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=flag == "train",
                            num_workers=self.args.num_workers, drop_last=flag == "train")
        print(flag, len(dataset))
        return dataset, loader

    def _target(self, batch_y):
        f_dim = -1 if self.args.features == "MS" else 0
        return batch_y.float().to(self.device)[:, -self.args.pred_len:, f_dim:]

    def _select_criterion(self):
        return nn.MSELoss()

    def train(self, setting):
        _, train_loader = self._get_data("train")
        _, val_loader = self._get_data("val")
        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        self.model.freeze_backbone(False)
        self.opt = optim.AdamW(self.model.backbone_parameters(), lr=self.args.learning_rate,
                               weight_decay=self.args.weight_decay)
        criterion = self._select_criterion()
        stopper = EarlyStopping(self.args.patience, verbose=True)
        for epoch in range(self.args.train_epochs):
            self.model.train()
            losses = []
            for batch_x, batch_y, _, _ in train_loader:
                x = batch_x.float().to(self.device)
                y = self._target(batch_y)
                self.opt.zero_grad()
                pred = self.model(x, backbone_only=True)
                loss = criterion(pred, y)
                loss.backward()
                self.opt.step()
                self.model.backbone.store_grad()
                losses.append(loss.item())
            val_loss = self.vali(None, val_loader, criterion)
            print("Epoch: {} | Train Loss: {:.6f} Vali Loss: {:.6f}".format(
                epoch + 1, float(np.mean(losses)), val_loss))
            stopper(val_loss, self.model, path)
            if stopper.early_stop:
                break
            adjust_learning_rate(self.opt, epoch + 1, self.args)
        self.model.load_state_dict(torch.load(os.path.join(path, "checkpoint.pth"), map_location=self.device))
        return self.model

    def load_pretrained(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch_x, batch_y, _, _ in vali_loader:
                pred = self.model(batch_x.float().to(self.device), backbone_only=True)
                losses.append(criterion(pred, self._target(batch_y)).item())
        return float(np.mean(losses))

    def _append_observation(self, x):
        if self.past is None:
            self.past = x.detach().clone()
        else:
            self.past = torch.cat([self.past, x[:, -1:, :].detach()], dim=1)
        max_len = self.args.dyname_past_num + self.args.seq_len + self.args.pred_len
        self.past = self.past[:, -max_len:]

    def _update_gate(self, x, y, past):
        self.model.train()
        self.opt_gate.zero_grad()
        updated = self.model(x, past)
        loss = self._select_criterion()(updated, y)
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.model.gate_parameters()), 1.0)
        self.opt_gate.step()
        self.model.update_signal(loss.item())

    def _update_gate_from_cache(self, cached):
        rep, stacked, blend, y = cached
        rep = rep.to(self.device)
        stacked = stacked.to(self.device)
        y = y.to(self.device)
        self.model.train()
        self.opt_gate.zero_grad()
        updated, _, _ = self.model.combine_cached(rep, stacked, blend=blend)
        loss = self._select_criterion()(updated, y)
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.model.gate_parameters()), 1.0)
        self.opt_gate.step()
        self.model.update_signal(loss.item())

    def _online_batch(self, batch_x, batch_y):
        x = batch_x.float().to(self.device)
        y = self._target(batch_y)
        self._append_observation(x)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x, self.past)
        if self.online != "none":
            self._update_gate(x, y, self.past)
        return rearrange(pred, "b t d -> b (t d)"), rearrange(y, "b t d -> b (t d)")

    def _delayed_online_batch(self, feedback_queue, batch_x, batch_y):
        batch_preds, batch_trues = [], []
        for i in range(batch_x.shape[0]):
            x = batch_x[i : i + 1].float().to(self.device)
            y = self._target(batch_y[i : i + 1])
            self._append_observation(x)

            if len(feedback_queue) >= self.args.pred_len:
                self._update_gate_from_cache(feedback_queue.popleft())

            self.model.eval()
            with torch.no_grad():
                pred, _, _, blend, rep, stacked = self.model(
                    x, self.past, return_details=True
                )
            feedback_queue.append((
                rep.detach().cpu(),
                stacked.detach().cpu(),
                float(blend),
                y.detach().cpu(),
            ))
            batch_preds.append(rearrange(pred, "b t d -> b (t d)"))
            batch_trues.append(rearrange(y, "b t d -> b (t d)"))

        return torch.cat(batch_preds, dim=0), torch.cat(batch_trues, dim=0)

    def test(self, setting):
        _, loader = self._get_data("test")
        self.model.freeze_backbone(True)
        self.opt_gate = optim.Adam(self.model.gate_parameters(), lr=self.args.dyname_online_lr)
        self.past = None
        preds, trues, maes, mses, rmses, mapes, mspes = [], [], [], [], [], [], []
        start = time.time()
        feedback_queue = deque() if self.args.delay_fb and self.online != "none" else None
        if feedback_queue is not None:
            print("[DELAY_FB] rolling origins; feedback delay={} steps".format(self.args.pred_len))
        for batch_x, batch_y, _, _ in tqdm(loader):
            if feedback_queue is not None:
                pred, true = self._delayed_online_batch(feedback_queue, batch_x, batch_y)
            else:
                pred, true = self._online_batch(batch_x, batch_y)
            preds.append(pred.detach().cpu())
            trues.append(true.detach().cpu())
            values = metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())
            for target, value in zip((maes, mses, rmses, mapes, mspes), values):
                target.append(value)
        pred_np = torch.cat(preds).numpy()
        true_np = torch.cat(trues).numpy()
        curves = [cumavg(x) for x in (maes, mses, rmses, mapes, mspes)]
        mae, mse, rmse, mape, mspe = [x[-1] for x in curves]
        elapsed = time.time() - start
        print("mse:{}, mae:{}, time:{}".format(mse, mae, elapsed))
        return [mae, mse, rmse, mape, mspe, elapsed], curves[0], curves[1], pred_np, true_np
