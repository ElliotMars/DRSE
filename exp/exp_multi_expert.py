import copy
import os
import time
import warnings
from collections import defaultdict
from collections import deque
from contextlib import contextmanager

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


class TS2VecEncoderWrapper(nn.Module):
    def __init__(self, encoder, mask):
        super().__init__()
        self.encoder = encoder
        self.mask = mask

    def forward(self, input):
        return self.encoder(input, mask=self.mask)


class ExpertNet(nn.Module):
    """
    Single FSNet expert built on the same structure used by run.sh.
    """

    expert_type = "fsnet"

    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.input_dim = args.enc_in + 7
        self.hidden_dim = 320
        self.output_dim = args.c_out * args.pred_len

        encoder = TSEncoder(
            input_dims=self.input_dim,
            output_dims=self.hidden_dim,
            hidden_dims=64,
            depth=10,
            device=self.device,
        )
        self.encoder = TS2VecEncoderWrapper(encoder, mask="all_true").to(self.device)
        self.regressor = nn.Linear(self.hidden_dim, self.output_dim).to(self.device)

    def _encode(self, x, x_mark):
        x_with_mark = torch.cat([x, x_mark], dim=-1)  # [B, T, enc_in + 7]
        # FSNet's dilated conv stack can fail on some CUDA/cuDNN combos;
        # run this block without cuDNN to keep training stable.
        with torch.backends.cudnn.flags(enabled=False):
            h = self.encoder(x_with_mark)  # [B, T, H]
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


class FSNetTimeExpertNet(nn.Module):
    """
    FSNet-Time expert built on the same structure used by run.sh.
    """

    expert_type = "fsnet_time"

    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.input_dim = args.seq_len
        self.hidden_dim = 320
        self.pred_len = args.pred_len
        self.output_dim = args.c_out * args.pred_len

        encoder = TSEncoder(
            input_dims=self.input_dim,
            output_dims=self.hidden_dim,
            hidden_dims=64,
            depth=10,
            device=self.device,
        )
        self.encoder_time = TS2VecEncoderWrapper(encoder, mask="all_true").to(self.device)
        self.regressor_time = nn.Linear(self.hidden_dim, self.pred_len).to(self.device)

    def _encode(self, x, x_mark):
        del x_mark
        # Matches exp_onenet_fsnet: call forward_time without an explicit mask.
        with torch.backends.cudnn.flags(enabled=False):
            h = self.encoder_time.encoder.forward_time(x)  # [B, C, H]
        return h

    def forward(self, x, x_mark, return_repr=False):
        h = self._encode(x, x_mark)
        y_hat = self.regressor_time(h).transpose(1, 2)  # [B, pred_len, C]
        y_hat = rearrange(y_hat, "b t d -> b (t d)")
        z = h.mean(dim=1)  # [B, H]
        if return_repr:
            return y_hat, z
        return y_hat

    def encode_feature(self, x, x_mark):
        h = self._encode(x, x_mark)
        return h.mean(dim=1)

    def store_grad(self):
        for name, layer in self.encoder_time.named_modules():
            if "PadConv" in type(layer).__name__:
                layer.store_grad()


class RoutingFeatureEncoder(nn.Module):
    """
    Shared feature encoder for routing. It produces a sequence feature from the
    original series; the router consumes the pooled feature to assign experts.
    """

    def __init__(self, args, hidden_dim, device):
        super().__init__()
        self.device = device
        self.input_dim = args.enc_in + 7
        self.input_proj = nn.Linear(self.input_dim, hidden_dim).to(self.device)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 2,
            dropout=args.dropout,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1).to(self.device)

    def forward(self, x, x_mark):
        x_with_mark = torch.cat([x, x_mark], dim=-1)  # [B, T, enc_in + 7]
        h = self.input_proj(x_with_mark)  # [B, T, H]
        h = h.transpose(0, 1)  # [T, B, H]
        h = self.encoder(h)
        h = h.transpose(0, 1)  # [B, T, H]
        return h, h[:, -1, :]


class net(nn.Module):
    """
    Multi-expert version with mixed FSNet and FSNet-Time experts.
    Output shape without router: [batch_size, num_experts, output_dim].
    """

    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.num_experts = max(1, int(getattr(args, "num_experts", 4)))
        self.top_k = max(1, min(int(getattr(args, "top_k", self.num_experts)), self.num_experts))
        self.num_time_experts = min(2, self.num_experts)
        self.num_fsnet_experts = self.num_experts - self.num_time_experts
        self.hidden_dim = 320
        self.c_out = args.c_out
        self.pred_len = args.pred_len
        self.router_temperature = max(float(getattr(args, "router_temperature", 2.0)), 1e-3)

        experts = []
        for _ in range(self.num_fsnet_experts):
            experts.append(ExpertNet(args, device=self.device))
        for _ in range(self.num_time_experts):
            experts.append(FSNetTimeExpertNet(args, device=self.device))
        self.experts = nn.ModuleList(experts)

        self.feature_encoder = RoutingFeatureEncoder(args, self.hidden_dim, self.device)
        self.router = nn.Linear(self.hidden_dim, self.c_out * self.num_experts).to(self.device)
        # Start from a safe uniform ensemble.
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)
        self.use_router = False

    def router_parameters(self):
        return list(self.feature_encoder.parameters()) + list(self.router.parameters())

    def set_router_mode(self, use_router: bool):
        self.use_router = use_router

    def _compute_gates(self, x, x_mark):
        _, route_feature = self.feature_encoder(x, x_mark)
        scores = self.router(route_feature)
        scores = scores.view(x.shape[0], self.c_out, self.num_experts)
        gates = torch.softmax(scores / self.router_temperature, dim=-1)

        if self.top_k < self.num_experts:
            _, topk_idx = torch.topk(gates, k=self.top_k, dim=-1)
            mask = torch.zeros_like(gates)
            mask.scatter_(2, topk_idx, 1.0)
            gates = gates * mask
            gates = gates / (gates.sum(dim=-1, keepdim=True) + 1e-8)
        return gates

    def _prepare_expert_input(self, expert, x, x_mark):
        # FSNet-Time applies masks in-place, so each expert gets an isolated input.
        del expert
        return x.clone(), x_mark

    def forward_experts(self, x, x_mark, return_repr=False):
        if return_repr:
            ys, zs = [], []
            for expert in self.experts:
                expert_x, expert_x_mark = self._prepare_expert_input(expert, x, x_mark)
                y_i, z_i = expert(expert_x, expert_x_mark, return_repr=True)
                ys.append(y_i)
                zs.append(z_i)
            outputs = torch.stack(ys, dim=1)  # [B, E, D]
            reps = torch.stack(zs, dim=1)  # [B, E, H]
            return outputs, reps
        expert_outputs = []
        for expert in self.experts:
            expert_x, expert_x_mark = self._prepare_expert_input(expert, x, x_mark)
            expert_outputs.append(expert(expert_x, expert_x_mark))
        outputs = torch.stack(expert_outputs, dim=1)  # [B, E, D]
        return outputs

    def aggregate_with_gates(self, gates, outputs):
        batch_size = outputs.shape[0]
        outputs = outputs.view(batch_size, self.num_experts, self.pred_len, self.c_out)
        weights = gates.permute(0, 2, 1).unsqueeze(2)
        prediction = torch.sum(weights * outputs, dim=1)
        return rearrange(prediction, "b t d -> b (t d)")

    def route_and_aggregate(self, x, x_mark, outputs=None):
        gates = self._compute_gates(x, x_mark)
        if outputs is None:
            outputs = self.forward_experts(x, x_mark, return_repr=False)
        return self.aggregate_with_gates(gates, outputs)

    def forward(self, x, x_mark):
        if self.use_router:
            return self.route_and_aggregate(x, x_mark)
        return self.forward_experts(x, x_mark, return_repr=False)

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
        self.lambda_div = float(getattr(args, "lambda_div", 0.0))
        self.tsb_alpha = float(getattr(args, "tsb_alpha", 0.5))
        self.tsb_eps = float(getattr(args, "tsb_eps", 1e-8))
        self.tsb_buffer_size = int(getattr(args, "tsb_buffer_size", 8))
        self.online_buffer = deque(maxlen=self.tsb_buffer_size)
        self.expert_params = list(self.model.experts.parameters())
        self.router_params = self.model.router_parameters()
        self.base_learning_rate_expert = float(self.args.online_lr_expert)
        self.base_learning_rate_router = float(self.args.online_lr_router)
        self.expert_grad_clip = float(getattr(args, "expert_grad_clip", 1.0))
        self.router_grad_clip = float(getattr(args, "router_grad_clip", 0.5))
        self.router_entropy_weight = float(getattr(args, "router_entropy_weight", 1e-3))
        self.online_log_interval = int(getattr(args, "online_log_interval", 500))
        self.online_step = 0
        self.fallback_count = 0
        self.fallback_channel_count = 0
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
        # A sudden error spike must make the update more conservative, not
        # increase the step size. Never exceed the configured online LR.
        lr_scale = float(np.clip(1.0 / np.sqrt(max(ratio, self.tsb_eps)), 0.1, 1.0))
        expert_lr = self.base_learning_rate_expert * lr_scale
        router_lr = self.base_learning_rate_router * lr_scale
        dynamic_tsb_alpha = float(np.clip(self.tsb_alpha / lr_scale, self.tsb_alpha, 0.95))
        return expert_lr, router_lr, dynamic_tsb_alpha

    def _update_online_mse_state(self, mse_value):
        mse_value = float(mse_value)
        self.prev_online_mse = mse_value
        if self.online_mse_ema is None:
            self.online_mse_ema = mse_value
        else:
            beta = self.online_mse_ema_beta
            self.online_mse_ema = beta * self.online_mse_ema + (1.0 - beta) * mse_value

    @contextmanager
    def _fsnet_state_updates(self, enabled):
        modules = [m for m in self.model.modules() if hasattr(m, "state_updates_enabled")]
        previous = [m.state_updates_enabled for m in modules]
        for module in modules:
            module.state_updates_enabled = enabled
        try:
            yield
        finally:
            for module, old_value in zip(modules, previous):
                module.state_updates_enabled = old_value


    def _safe_prediction(self, gates, outputs):
        """Use router output unless an individual value is NaN or infinite.

        Expert disagreement is expected under distribution shift and must be
        resolved by the learned router. The old sample-wise median fallback
        bypassed all channel gates when one ECL channel shifted, so fallback is
        now limited to non-finite values and is applied coordinate-wise.
        """
        raw_prediction = self.model.aggregate_with_gates(gates, outputs)
        finite_experts = torch.isfinite(outputs)
        finite_sum = torch.where(
            finite_experts, outputs, torch.zeros_like(outputs)
        ).sum(dim=1)
        finite_count = finite_experts.sum(dim=1).clamp_min(1)
        finite_mean = finite_sum / finite_count

        unsafe_values = ~torch.isfinite(raw_prediction)
        safe_prediction = torch.where(unsafe_values, finite_mean, raw_prediction)
        safe_prediction = torch.nan_to_num(
            safe_prediction, nan=0.0, posinf=0.0, neginf=0.0
        )

        unsafe_channels = unsafe_values.view(
            outputs.shape[0], self.args.pred_len, self.args.c_out
        ).any(dim=1)
        self.fallback_count += int(unsafe_values.sum().item())
        self.fallback_channel_count += int(unsafe_channels.sum().item())
        return safe_prediction, unsafe_channels

    @staticmethod
    def _independent_expert_mse(expert_outputs, target):
        """Sum independently supervised expert MSEs, matching OneNet's scale."""
        per_expert = (expert_outputs - target.unsqueeze(1)).pow(2).mean(dim=2)
        return per_expert.sum(dim=1).mean()

    def load_pretrained(self, checkpoint_path):
        state = torch.load(checkpoint_path, map_location=self.device)
        try:
            self.model.load_state_dict(state)
        except RuntimeError as exc:
            raise RuntimeError(
                "Incompatible multi-expert checkpoint. Re-run the v2 pretraining script."
            ) from exc
        self.model.set_router_mode(True)
        self.opt_expert = self._select_expert_optimizer()
        self.opt_router = self._select_router_optimizer()
        optimizer_path = os.path.join(os.path.dirname(checkpoint_path), "optimizer.pth")
        if os.path.exists(optimizer_path):
            optimizer_state = torch.load(optimizer_path, map_location=self.device)
            self.opt_expert.load_state_dict(optimizer_state["expert"])
            self.opt_router.load_state_dict(optimizer_state["router"])
            print("Loaded optimizer state:", optimizer_path)
        else:
            print("Optimizer state not found; starting online moments from zero")
        self._set_optimizer_lr(self.opt_expert, self.base_learning_rate_expert)
        self._set_optimizer_lr(self.opt_router, self.base_learning_rate_router)
        return self.model

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
            delay_fb=args.delay_fb,
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
        self.opt_expert = optim.AdamW(
            self.expert_params, lr=self.args.learning_rate_expert, weight_decay=self.args.weight_decay
        )
        return self.opt_expert

    def _select_router_optimizer(self):
        self.opt_router = optim.AdamW(
            self.router_params, lr=self.args.learning_rate_router, weight_decay=self.args.weight_decay
        )
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

        self.model.set_router_mode(True)
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
                self.opt_router.zero_grad()

                expert_outputs, routed_pred, expert_reps, gates, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark
                )
                expert_losses = [criterion(expert_outputs[:, e, :], true) for e in range(expert_outputs.shape[1])]
                loss_pred = torch.stack(expert_losses).sum()

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

                loss_expert = loss_pred + self.lambda_div * loss_div
                gate_entropy = -(gates.clamp_min(1e-8) * gates.clamp_min(1e-8).log()).sum(dim=-1).mean()
                loss_router = criterion(routed_pred, true) - self.router_entropy_weight * gate_entropy
                loss = loss_expert + loss_router
                train_loss.append(loss_expert.item())

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss_expert.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.opt_expert)
                    scaler.unscale_(self.opt_router)
                    nn.utils.clip_grad_norm_(self.expert_params, self.expert_grad_clip)
                    nn.utils.clip_grad_norm_(self.router_params, self.router_grad_clip)
                    scaler.step(self.opt_expert)
                    scaler.step(self.opt_router)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.expert_params, self.expert_grad_clip)
                    nn.utils.clip_grad_norm_(self.router_params, self.router_grad_clip)
                    self.opt_expert.step()
                    self.opt_router.step()
                self.model.store_grad()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.4f} Vali Loss: {3:.4f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss
                )
            )
            previous_best = early_stopping.best_score
            early_stopping(vali_loss, self.model, path)
            improved = previous_best is None or -vali_loss >= previous_best + early_stopping.delta
            if improved:
                torch.save(
                    {"expert": self.opt_expert.state_dict(), "router": self.opt_router.state_dict()},
                    os.path.join(path, "optimizer.pth"),
                )
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(self.opt_expert, epoch + 1, self.args)
            adjust_learning_rate(self.opt_router, epoch + 1, self.args)

        best_model_path = path + "/checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))
        optimizer_path = os.path.join(path, "optimizer.pth")
        if os.path.exists(optimizer_path):
            optimizer_state = torch.load(optimizer_path, map_location=self.device)
            self.opt_expert.load_state_dict(optimizer_state["expert"])
            self.opt_router.load_state_dict(optimizer_state["router"])
        self._set_optimizer_lr(self.opt_expert, self.base_learning_rate_expert)
        self._set_optimizer_lr(self.opt_router, self.base_learning_rate_router)
        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        self.model.set_router_mode(True)
        total_loss = []
        with torch.no_grad(), self._fsnet_state_updates(False):
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                _, routed_pred, _, _, true = self._process_one_batch(
                    vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark, mode="vali"
                )
                loss = criterion(routed_pred.detach().cpu(), true.detach().cpu())
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def _delayed_online_batch(
        self, dataset, feedback_queue, batch_x, batch_y, batch_x_mark, batch_y_mark
    ):
        batch_preds, batch_trues = [], []
        for i in range(batch_x.shape[0]):
            if len(feedback_queue) >= self.args.pred_len:
                released = feedback_queue.popleft()
                self._ol_one_batch(dataset, *released)

            current = (
                batch_x[i : i + 1].detach().clone(),
                batch_y[i : i + 1].detach().clone(),
                batch_x_mark[i : i + 1].detach().clone(),
                batch_y_mark[i : i + 1].detach().clone(),
            )
            pred, true = self._predict_without_update(
                current[0], current[1], current[2]
            )
            feedback_queue.append(current)
            batch_preds.append(pred)
            batch_trues.append(true)

        return torch.cat(batch_preds, dim=0), torch.cat(batch_trues, dim=0)

    def test(self, setting):
        test_data, test_loader = self._get_data(flag="test")

        self.model.eval()
        self.model.set_router_mode(True)
        if self.online == "regressor":
            for expert in self.model.experts:
                for name, parameter in expert.named_parameters():
                    if "regressor" not in name:
                        parameter.requires_grad = False

        preds = []
        trues = []
        start = time.time()
        maes, mses, rmses, mapes, mspes = [], [], [], [], []
        feedback_queue = deque() if self.args.delay_fb and self.online != "none" else None
        if feedback_queue is not None:
            print("[DELAY_FB] rolling origins; feedback delay={} steps".format(self.args.pred_len))
        for batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(test_loader):
            if feedback_queue is not None:
                pred, true = self._delayed_online_batch(
                    test_data, feedback_queue, batch_x, batch_y, batch_x_mark, batch_y_mark
                )
            else:
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
            if self.online == "none":
                return self._predict_without_update(batch_x, batch_y, batch_x_mark)
            return self._ol_one_batch(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark)

        x = batch_x.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y = batch_y.float()

        gates = self.model._compute_gates(x, batch_x_mark)
        expert_outputs, expert_reps = self.model.forward_experts(x, batch_x_mark, return_repr=True)
        routed_pred = self.model.aggregate_with_gates(gates, expert_outputs.detach())
        f_dim = -1 if self.args.features == "MS" else 0
        batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
        true = rearrange(batch_y, "b t d -> b (t d)")
        return expert_outputs, routed_pred, expert_reps, gates, true

    def _predict_without_update(self, batch_x, batch_y, batch_x_mark):
        x = batch_x.float().to(self.device)
        x_mark = batch_x_mark.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        f_dim = -1 if self.args.features == "MS" else 0
        batch_y = batch_y[:, -self.args.pred_len :, f_dim:]
        true = rearrange(batch_y, "b t d -> b (t d)")
        with torch.no_grad(), self._fsnet_state_updates(False):
            gates = self.model._compute_gates(x, x_mark)
            outputs = self.model.forward_experts(x, x_mark, return_repr=False)
            prediction, _ = self._safe_prediction(gates, outputs)
        return prediction, true

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
        # TSB remains sample-wise, while its reference gradient is evaluated in
        # one read-only buffer batch to avoid advancing FSNet state repeatedly.
        for i in range(x.shape[0]):
            x_t = x[i : i + 1]
            x_mark_t = batch_x_mark[i : i + 1]
            y_t = true[i : i + 1]

            pred_t = None
            for inner_idx in range(self.n_inner):
                expert_lr, router_lr, dynamic_tsb_alpha = self._adaptive_online_hparams()
                self._set_optimizer_lr(self.opt_expert, expert_lr)
                self._set_optimizer_lr(self.opt_router, router_lr)

                # Step A: pre-update prediction and current expert gradient.
                self.opt_expert.zero_grad()
                self.opt_router.zero_grad()
                with torch.no_grad():
                    gates_t = self.model._compute_gates(x_t, x_mark_t)
                expert_outputs_t = self.model.forward_experts(x_t, x_mark_t, return_repr=False)
                if inner_idx == 0:
                    raw_pred_t = self.model.aggregate_with_gates(
                        gates_t, expert_outputs_t.detach()
                    )
                    pred_t, unsafe_channels_t = self._safe_prediction(
                        gates_t, expert_outputs_t.detach()
                    )
                loss_cur = self._independent_expert_mse(expert_outputs_t, y_t)
                loss_cur.backward()
                g_cur = [
                    p.grad.detach().clone() if p.grad is not None else None
                    for p in self.expert_params
                ]

                # Step B: a batched, read-only TSB reference gradient.
                g_ref = [torch.zeros_like(p, device=p.device) for p in self.expert_params]
                if len(self.online_buffer) > 0:
                    x_b = torch.cat([item[0] for item in self.online_buffer], dim=0)
                    x_mark_b = torch.cat([item[1] for item in self.online_buffer], dim=0)
                    y_b = torch.cat([item[2] for item in self.online_buffer], dim=0)
                    self.opt_expert.zero_grad()
                    self.opt_router.zero_grad()
                    with self._fsnet_state_updates(False):
                        expert_outputs_b = self.model.forward_experts(
                            x_b, x_mark_b, return_repr=False
                        )
                        loss_b = self._independent_expert_mse(expert_outputs_b, y_b)
                    loss_b.backward()
                    g_ref = [
                        p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                        for p in self.expert_params
                    ]

                # Step C + D: EMA smoothing and TSB conflict projection.
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

                # Step E1: clipped expert update.
                expert_grad_norm = nn.utils.clip_grad_norm_(self.expert_params, self.expert_grad_clip)
                if torch.isfinite(expert_grad_norm):
                    self.opt_expert.step()
                    self.model.store_grad()
                else:
                    print("[ONLINE] skipped non-finite expert update at step", self.online_step)
                self.opt_expert.zero_grad()
                self.opt_router.zero_grad()

                # Step E2: router-only update. The extra expert forward is
                # read-only so it cannot advance q_ema/trigger/memory.
                with torch.no_grad(), self._fsnet_state_updates(False):
                    outputs_detached = self.model.forward_experts(
                        x_t, x_mark_t, return_repr=False
                    ).detach()
                gates_router = self.model._compute_gates(x_t, x_mark_t)
                y_hat_router = self.model.aggregate_with_gates(gates_router, outputs_detached)
                gate_entropy = -(
                    gates_router.clamp_min(1e-8) * gates_router.clamp_min(1e-8).log()
                ).sum(dim=-1).mean()
                loss_router = criterion(y_hat_router, y_t)
                loss_router.backward()
                router_grad_norm = nn.utils.clip_grad_norm_(self.router_params, self.router_grad_clip)
                if torch.isfinite(router_grad_norm):
                    self.opt_router.step()
                else:
                    print("[ONLINE] skipped non-finite router update at step", self.online_step)
                self.opt_router.zero_grad()
                self.opt_expert.zero_grad()

                online_mse = criterion(y_hat_router.detach(), y_t).item()
                self._update_online_mse_state(online_mse)

                if self.online_log_interval > 0 and self.online_step % self.online_log_interval == 0:
                    expert_mse = (expert_outputs_t.detach() - y_t.unsqueeze(1)).pow(2).mean(dim=2)
                    uniform_mse = criterion(expert_outputs_t.detach().mean(dim=1), y_t).item()
                    pred_by_channel = (
                        pred_t.view(-1, self.args.pred_len, self.args.c_out)
                        - y_t.view(-1, self.args.pred_len, self.args.c_out)
                    ).pow(2).mean(dim=1)
                    worst_channel_mse, worst_channel = pred_by_channel.max(dim=1)
                    raw_mse = criterion(raw_pred_t, y_t).item()
                    print(
                        "[ONLINE] step={} mse={:.6f} raw={:.6f} uniform={:.6f} experts={} "
                        "worst_ch={}:{} unsafe_ch={} "
                        "lr_e={:.2e} lr_r={:.2e} alpha={:.3f} entropy={:.3f} "
                        "grad_e={:.3f} grad_r={:.3f} "
                        "fallback_values={} fallback_channels={}".format(
                            self.online_step,
                            criterion(pred_t, y_t).item(),
                            raw_mse,
                            uniform_mse,
                            [round(v, 6) for v in expert_mse[0].tolist()],
                            int(worst_channel[0].item()),
                            round(float(worst_channel_mse[0].item()), 6),
                            int(unsafe_channels_t[0].sum().item()),
                            expert_lr,
                            router_lr,
                            dynamic_tsb_alpha,
                            gate_entropy.item(),
                            float(expert_grad_norm),
                            float(router_grad_norm),
                            self.fallback_count,
                            self.fallback_channel_count,
                        )
                    )

            self.online_buffer.append(
                (x_t.detach().clone(), x_mark_t.detach().clone(), y_t.detach().clone())
            )
            preds.append(pred_t.detach())
            trues.append(y_t.detach())
            self.online_step += 1

        return torch.cat(preds, dim=0), torch.cat(trues, dim=0)
