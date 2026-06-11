import copy
import os
import time
import warnings
from collections import defaultdict
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.data_loader import Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.ts2vec.fsnet import TSEncoder
from utils.metrics import cumavg, metric
from utils.tools import EarlyStopping, adjust_learning_rate

warnings.filterwarnings("ignore")


class ExpertNet(nn.Module):
    """
    Single expert built on FSNet (TSEncoder) backbone and linear head.
    """

    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.input_dim = args.enc_in + 7
        self.hidden_dim = 320
        self.output_dim = args.c_out * args.pred_len

        self.encoder = TSEncoder(
            input_dims=self.input_dim,
            output_dims=self.hidden_dim,
            hidden_dims=64,
            depth=10,
            device=self.device,
        ).to(self.device)
        self.regressor = nn.Linear(self.hidden_dim, self.output_dim).to(self.device)

    def _encode(self, x, x_mark):
        x_with_mark = torch.cat([x, x_mark], dim=-1)  # [B, T, enc_in + 7]
        # FSNet's dilated conv stack can fail on some CUDA/cuDNN combos;
        # run this block without cuDNN to keep training stable.
        with torch.backends.cudnn.flags(enabled=False):
            h = self.encoder(x_with_mark, mask="all_true")  # [B, T, H]
        return h

    def forward(self, x, x_mark, return_repr=False):
        h = self._encode(x, x_mark)
        z = h.mean(dim=1)  # [B, H]
        y_hat = self.regressor(h[:, -1, :])  # [B, D]
        if return_repr:
            return y_hat, z
        return y_hat

    def encode_feature(self, x, x_mark):
        h = self._encode(x, x_mark)
        return h[:, -1, :]

    def store_grad(self):
        for name, layer in self.encoder.named_modules():
            if "PadConv" in type(layer).__name__:
                layer.store_grad()


class net(nn.Module):
    """
    Multi-expert version: no OCP, no expert aggregation in forward.
    Output shape: [num_experts, batch_size, output_dim].
    """

    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.num_experts = max(1, int(getattr(args, "num_experts", 4)))
        self.top_k = max(1, min(int(getattr(args, "top_k", 2)), self.num_experts))

        base_expert = ExpertNet(args, device=self.device)
        self.hidden_dim = base_expert.hidden_dim
        self.router_input_dim = args.enc_in + 7
        self.router_input_proj = nn.Linear(self.router_input_dim, self.hidden_dim).to(self.device)
        router_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=8,
            dim_feedforward=self.hidden_dim * 2,
            dropout=args.dropout,
            activation="gelu",
        )
        self.router_encoder = nn.TransformerEncoder(router_layer, num_layers=1).to(self.device)
        self.experts = nn.ModuleList([copy.deepcopy(base_expert) for _ in range(self.num_experts)])
        self.router = nn.Linear(self.hidden_dim, self.num_experts).to(self.device)
        self.use_router = False

    def set_router_mode(self, use_router: bool):
        self.use_router = use_router

    def _compute_gates(self, x, x_mark):
        # h: [batch_size, hidden_dim], from a lightweight shared router encoder.
        x_with_mark = torch.cat([x, x_mark], dim=-1)  # [B, T, enc_in + 7]
        h = self.router_input_proj(x_with_mark)  # [B, T, H]
        h = h.transpose(0, 1)  # [T, B, H]
        h = self.router_encoder(h)  # [T, B, H]
        h = h.transpose(0, 1)[:, -1, :]  # [B, H]
        scores = self.router(h)  # [batch_size, num_experts]
        gates = torch.softmax(scores, dim=-1)

        topk_vals, topk_idx = torch.topk(gates, k=self.top_k, dim=-1)
        del topk_vals
        mask = torch.zeros_like(gates)
        mask.scatter_(1, topk_idx, 1.0)
        gates = gates * mask
        gates = gates / (gates.sum(dim=-1, keepdim=True) + 1e-8)
        return gates

    def forward_experts(self, x, x_mark, return_repr=False):
        if return_repr:
            ys, zs = [], []
            for expert in self.experts:
                y_i, z_i = expert(x, x_mark, return_repr=True)
                ys.append(y_i)
                zs.append(z_i)
            outputs = torch.stack(ys, dim=1)  # [B, E, D]
            reps = torch.stack(zs, dim=1)  # [B, E, H]
            return outputs, reps
        expert_outputs = [expert(x, x_mark) for expert in self.experts]
        outputs = torch.stack(expert_outputs, dim=1)  # [B, E, D]
        return outputs

    def route_and_aggregate(self, x, x_mark, outputs=None):
        if outputs is None:
            outputs = self.forward_experts(x, x_mark, return_repr=False)
        gates = self._compute_gates(x, x_mark)
        y_hat = torch.sum(gates.unsqueeze(-1) * outputs, dim=1)
        return y_hat

    def forward(self, x, x_mark):
        outputs = self.forward_experts(x, x_mark, return_repr=False)
        if not self.use_router:
            return outputs
        return self.route_and_aggregate(x, x_mark, outputs=outputs)

    def store_grad(self):
        for expert in self.experts:
            expert.store_grad()


class Exp_TS2VecSupervised(Exp_Basic):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.online = args.online_learning
        assert self.online in ["none", "full", "regressor"]
        self.n_inner = args.n_inner
        self.opt_str = args.opt
        self.model = net(args, device=self.device)
        self.lambda_div = float(getattr(args, "lambda_div", 0.05))
        self.tsb_alpha = float(getattr(args, "tsb_alpha", 0.5))
        self.tsb_eps = float(getattr(args, "tsb_eps", 1e-8))
        self.tsb_buffer_size = int(getattr(args, "tsb_buffer_size", 32))
        self.online_buffer = deque(maxlen=self.tsb_buffer_size)
        self.expert_params = list(self.model.experts.parameters())
        self.router_params = list(self.model.router.parameters())
        self.base_learning_rate_expert = float(self.args.learning_rate_expert)
        self.base_learning_rate_router = float(self.args.learning_rate_router)
        self.prev_online_mse = None
        self.online_mse_ema = None
        self.online_mse_ema_beta = 0.9

    def _set_optimizer_lr(self, optimizer, lr):
        for group in optimizer.param_groups:
            group["lr"] = lr

    def _adaptive_online_hparams(self):
        if self.prev_online_mse is None or self.online_mse_ema is None:
            return self.base_learning_rate_expert, self.base_learning_rate_router, self.tsb_alpha

        ratio = self.prev_online_mse / (self.online_mse_ema + self.tsb_eps)
        lr_scale = float(np.clip(np.sqrt(ratio), 0.5, 2.0))
        expert_lr = self.base_learning_rate_expert * lr_scale
        router_lr = self.base_learning_rate_router * lr_scale
        dynamic_tsb_alpha = float(np.clip(self.tsb_alpha / lr_scale, 0.05, 0.95))
        return expert_lr, router_lr, dynamic_tsb_alpha

    def _update_online_mse_state(self, mse_value):
        mse_value = float(mse_value)
        self.prev_online_mse = mse_value
        if self.online_mse_ema is None:
            self.online_mse_ema = mse_value
        else:
            beta = self.online_mse_ema_beta
            self.online_mse_ema = beta * self.online_mse_ema + (1.0 - beta) * mse_value

    def _get_data(self, flag):
        args = self.args

        data_dict_ = {
            "ETTh1": Dataset_ETT_hour,
            "ETTh2": Dataset_ETT_hour,
            "ETTm1": Dataset_ETT_minute,
            "ETTm2": Dataset_ETT_minute,
            "WTH": Dataset_Custom,
            "ECL": Dataset_Custom,
            "Solar": Dataset_Custom,
            "custom": Dataset_Custom,
        }
        data_dict = defaultdict(lambda: Dataset_Custom, data_dict_)
        Data = data_dict[self.args.data]
        timeenc = 2

        if flag == "test":
            shuffle_flag = False
            drop_last = False
            batch_size = args.test_bsz
            freq = args.freq
        elif flag == "val":
            shuffle_flag = False
            drop_last = False
            batch_size = args.batch_size
            freq = args.detail_freq
        elif flag == "pred":
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            freq = args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
        )

        return data_set, data_loader

    def _select_expert_optimizer(self):
        self.opt_expert = optim.AdamW(self.expert_params, lr=self.args.learning_rate_expert)
        return self.opt_expert

    def _select_router_optimizer(self):
        self.opt_router = optim.AdamW(self.router_params, lr=self.args.learning_rate_router)
        return self.opt_router

    def _select_criterion(self):
        return nn.MSELoss()

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        self.model.set_router_mode(False)
        self.opt_expert = self._select_expert_optimizer()
        self.opt_router = self._select_router_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                self.opt_expert.zero_grad()

                expert_preds, expert_reps, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark
                )
                # Prediction loss across all experts.
                target = true.unsqueeze(1).expand_as(expert_preds)  # [B, E, D]
                loss_pred = criterion(expert_preds, target) * expert_preds.shape[1]

                # Diversity loss across experts for same input batch.
                z = F.normalize(expert_reps, p=2, dim=-1, eps=1e-8)  # [B, E, H]
                sims = []
                e_num = z.shape[1]
                for ii in range(e_num):
                    for jj in range(e_num):
                        if ii == jj:
                            continue
                        sim_ij = F.cosine_similarity(z[:, ii, :], z[:, jj, :], dim=-1).mean()
                        sims.append(sim_ij)
                if len(sims) > 0:
                    loss_div = torch.stack(sims).mean()
                else:
                    loss_div = torch.tensor(0.0, device=self.device)

                loss = loss_pred + self.lambda_div * loss_div
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(self.opt_expert)
                    scaler.update()
                else:
                    loss.backward()
                    self.opt_expert.step()
                self.model.store_grad()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.4f} Vali Loss: {3:.4f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss
                )
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(self.opt_expert, epoch + 1, self.args)

        best_model_path = path + "/checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        self.model.set_router_mode(False)
        total_loss = []
        for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
            expert_preds, _, true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark, mode="vali"
            )
            target = true.unsqueeze(1).expand_as(expert_preds)  # [B, E, D]
            loss = criterion(expert_preds.detach().cpu(), target.detach().cpu()) * expert_preds.shape[1]
            total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting):
        test_data, test_loader = self._get_data(flag="test")

        self.model.eval()
        self.model.set_router_mode(True)

        preds = []
        trues = []
        start = time.time()
        maes, mses, rmses, mapes, mspes = [], [], [], [], []
        for batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(test_loader):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark, mode="test"
            )
            preds.append(pred.detach().cpu())
            trues.append(true.detach().cpu())
            mae, mse, rmse, mape, mspe = metric(
                pred.detach().cpu().numpy(), true.detach().cpu().numpy()
            )
            maes.append(mae)
            mses.append(mse)
            rmses.append(rmse)
            mapes.append(mape)
            mspes.append(mspe)

        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()
        print("test shape:", preds.shape, trues.shape)

        MAE, MSE, RMSE, MAPE, MSPE = (
            cumavg(maes),
            cumavg(mses),
            cumavg(rmses),
            cumavg(mapes),
            cumavg(mspes),
        )
        mae, mse, rmse, mape, mspe = MAE[-1], MSE[-1], RMSE[-1], MAPE[-1], MSPE[-1]

        exp_time = time.time() - start
        print("mse:{}, mae:{}, time:{}".format(mse, mae, exp_time))
        return [mae, mse, rmse, mape, mspe, exp_time], MAE, MSE, preds, trues

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode="train"):
        if mode == "test":
            return self._ol_one_batch(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark)

        x = batch_x.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y = batch_y.float()

        expert_outputs, expert_reps = self.model.forward_experts(x, batch_x_mark, return_repr=True)
        f_dim = -1 if self.args.features == "MS" else 0
        batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
        true = rearrange(batch_y, "b t d -> b (t d)")
        return expert_outputs, expert_reps, true

    def _ol_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        criterion = self._select_criterion()
        x = batch_x.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        f_dim = -1 if self.args.features == "MS" else 0
        batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
        true = rearrange(batch_y, "b t d -> b (t d)")

        preds = []
        trues = []
        # Online TSB is sample-wise (batch_size=1 is expected).
        for i in range(x.shape[0]):
            x_t = x[i : i + 1]
            x_mark_t = batch_x_mark[i : i + 1]
            y_t = true[i : i + 1]

            for _ in range(self.n_inner):
                expert_lr, router_lr, dynamic_tsb_alpha = self._adaptive_online_hparams()
                self._set_optimizer_lr(self.opt_expert, expert_lr)
                self._set_optimizer_lr(self.opt_router, router_lr)

                # Step A: current gradient for experts (router frozen by optimizer separation).
                self.opt_expert.zero_grad()
                self.opt_router.zero_grad()
                y_hat_t = self.model.route_and_aggregate(x_t, x_mark_t)
                loss_cur = criterion(y_hat_t, y_t)
                loss_cur.backward()
                g_cur = [
                    p.grad.detach().clone() if p.grad is not None else None
                    for p in self.expert_params
                ]

                # Step B: reference gradient from buffer
                g_ref = []
                for p in self.expert_params:
                    g_ref.append(torch.zeros_like(p, device=p.device))

                if len(self.online_buffer) > 0:
                    for x_b, x_mark_b, y_b in self.online_buffer:
                        self.opt_expert.zero_grad()
                        self.opt_router.zero_grad()
                        y_hat_b = self.model.route_and_aggregate(x_b, x_mark_b)
                        loss_b = criterion(y_hat_b, y_b)
                        loss_b.backward()
                        for j, p in enumerate(self.expert_params):
                            if p.grad is not None:
                                g_ref[j] += p.grad.detach()

                    inv_n = 1.0 / float(len(self.online_buffer))
                    for j in range(len(g_ref)):
                        g_ref[j] = g_ref[j] * inv_n

                # Step C + D: EMA smoothing + TSB filtering
                self.opt_expert.zero_grad()
                self.opt_router.zero_grad()
                for j, p in enumerate(self.expert_params):
                    if g_cur[j] is None:
                        continue
                    if len(self.online_buffer) == 0:
                        g_filtered = g_cur[j]
                    else:
                        g_smooth = (1.0 - dynamic_tsb_alpha) * g_cur[j] + dynamic_tsb_alpha * g_ref[j]
                        dot = torch.sum(g_smooth * g_ref[j])
                        if dot < 0:
                            ref_norm_sq = torch.sum(g_ref[j] * g_ref[j]) + self.tsb_eps
                            g_filtered = g_smooth - (dot / ref_norm_sq) * g_ref[j]
                        else:
                            g_filtered = g_smooth
                    p.grad = g_filtered.clone()

                # Step E1: update experts with TSB-filtered grads.
                self.opt_expert.step()
                self.model.store_grad()
                self.opt_expert.zero_grad()
                self.opt_router.zero_grad()

                # Step E2: router-only update (current sample only, no buffer, no TSB).
                with torch.no_grad():
                    outputs_detached = self.model.forward_experts(x_t, x_mark_t, return_repr=False).detach()
                self.opt_router.zero_grad()
                y_hat_router = self.model.route_and_aggregate(x_t, x_mark_t, outputs=outputs_detached)
                loss_router = criterion(y_hat_router, y_t)
                loss_router.backward()
                self.opt_router.step()
                self.opt_router.zero_grad()
                self.opt_expert.zero_grad()
                self._update_online_mse_state(loss_router.detach().item())

            # Step F: update online buffer
            self.online_buffer.append(
                (x_t.detach().clone(), x_mark_t.detach().clone(), y_t.detach().clone())
            )
            preds.append(y_hat_router.detach())
            trues.append(y_t.detach())

        return torch.cat(preds, dim=0), torch.cat(trues, dim=0)
