"""
Untargeted LAFEAT Attack.

Reference:
    https://github.com/fra31/auto-attack/blob/master/autoattack/autopgd_pt.py
"""
import time

import numpy as np
import torch
from torch import nn
import math


class LafeatAttack():
    def __init__(self, model, n_iter=100, norm='Linf', eps=None,
                 loss='ce', eot_iter=1, rho=.75, verbose=False,
                 device='cuda'):
        self.model = model
        self.n_iter = n_iter
        self.eps = eps - 1.9 * 1e-8
        self.norm = norm
        self.loss = loss
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.verbose = verbose
        self.device = device
        self.scale = True
        self.linear = True

    def dlr_loss(self, x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()

        return -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)) / (
                x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

    def check_right_index(self, output, labels):
        output_index = output.argmax(dim=-1) == labels
        mask = output_index.to(dtype=torch.int8)
        mask = torch.unsqueeze(mask, -1)
        return mask

    def get_output_scale(self, output):
        std_max_out = []
        maxk = max((10,))
        pred_val_out, pred_id_out = output.topk(maxk, 1, True, True)
        std_max_out.extend((pred_val_out[:, 0] - pred_val_out[:, 1] + 1e-8).cpu().numpy())
        scale_list = [item / 1.0 for item in std_max_out]
        scale_list = torch.tensor(scale_list).to(self.device)
        scale_list = torch.unsqueeze(scale_list, -1)
        return scale_list

    def attack_single_run(self, x_in, y_in):
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)

        if self.norm == 'Linf':
            t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * t / (
                t.reshape([t.shape[0], -1]).abs().max(dim=1, keepdim=True)[0].reshape([-1, 1, 1, 1]))


        elif self.norm == 'L2':
            t = torch.randn(x.shape).to(self.device).detach()
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * t / (
                    (t ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
        x_adv = x_adv.clamp(0., 1.)
        x_best_adv = x_adv.clone()

        if self.loss == 'ce':
            criterion_indiv = nn.CrossEntropyLoss(reduce=False, reduction='none')
        elif self.loss == 'dlr':
            criterion_indiv = self.dlr_loss
        else:
            raise ValueError('unknowkn loss')
        x_adv.requires_grad_()

        acc = self.model(x_adv)[-1].max(1)[1] == y

        step_size_begin = self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * torch.Tensor(
            [2.0]).to(
            self.device).detach().reshape([1, 1, 1, 1])
        x_adv_old = x_adv.clone()

        for i in range(self.n_iter):
            step_size = step_size_begin * (1 + math.cos(i / self.n_iter * math.pi)) * 0.5
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)

            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    x_adv_input = x_adv
                    outputs = self.model(x_adv_input)  # 1 forward pass (eot_iter = 1)
                    out_adv_256_15, out_adv_256_16, out_adv_256_17, out_adv_256_18, out_adv = outputs[-5], outputs[-4], \
                                                                                              outputs[-3], outputs[-2], \
                                                                                              outputs[-1]

                    mask_out_adv = self.check_right_index(out_adv, y)
                    mask_out_adv_grad = torch.unsqueeze(torch.unsqueeze(mask_out_adv.clone(), -1), -1)  # #############
                    mask_256_17 = self.check_right_index(out_adv_256_17, y)
                    scale_17 = self.scale_17
                    scale_output = self.get_output_scale(out_adv.clone().detach())
                    scale_output_17 = self.get_output_scale(out_adv_256_17.clone().detach())

                    if self.scale:
                        logits_prev = scale_17 * out_adv_256_17 * mask_256_17 / scale_output_17 + \
                                      (1 - scale_17 * mask_256_17) * out_adv / scale_output

                    elif not self.scale:
                        logits_prev = scale_17 * out_adv_256_17 * mask_256_17 + (1 - scale_17 * mask_256_17) * out_adv
                    loss_indiv_prev = criterion_indiv(logits_prev, y)
                    loss_prev = loss_indiv_prev.sum()
                    logits = out_adv

                grad += torch.autograd.grad(loss_prev, [x_adv])[0].detach()  # 1 backward pass (eot_iter = 1)

            grad /= float(self.eot_iter)
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                a = 0.75 if i > 0 else 1.0

                if self.norm == 'Linf':
                    x_adv_1 = x_adv + mask_out_adv_grad * step_size * torch.sign(grad)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, x - self.eps), x + self.eps), 0.0, 1.0)

                    x_adv_1 = torch.clamp(torch.min(
                        torch.max(x_adv + mask_out_adv_grad * ((x_adv_1 - x_adv) * a + grad2 * (1 - a)),
                                  x - self.eps),
                        x + self.eps), 0.0, 1.0)

                elif self.norm == 'L2':
                    x_adv_1 = x_adv + step_size * grad / ((grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (
                            ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        self.eps * torch.ones(x.shape).to(self.device).detach(),
                        ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()), 0.0, 1.0)
                    x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (
                            ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        self.eps * torch.ones(x.shape).to(self.device).detach(),
                        ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

                x_adv = x_adv_1 + 0.

            pred = logits.detach().max(1)[1] == y
            acc = torch.min(acc, pred)

            x_best_adv[(pred == 0).nonzero().squeeze()] = x_adv[(pred == 0).nonzero().squeeze()] + 0.

        return acc, x_best_adv

    def perturb(self, x_in, y_in, scale_17=0):
        self.scale_17 = scale_17
        self.seed = np.int(scale_17*10)
        assert self.norm in ['Linf', 'L2']
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)

        adv = x.clone()
        x_input = x
        acc = self.model(x_input)[-1].max(1)[1] == y
        if self.verbose:
            print('-------------------------- running {}-attack with epsilon {:.4f} --------------------------'.format(
                self.norm, self.eps))
            print('initial accuracy: {:.2%}'.format(acc.float().mean()))
        startt = time.time()

        torch.random.manual_seed(self.seed)
        torch.cuda.random.manual_seed(self.seed)

        ind_to_fool = acc.nonzero().squeeze()
        if len(ind_to_fool.shape) == 0: ind_to_fool = ind_to_fool.unsqueeze(0)
        if ind_to_fool.numel() != 0:
            x_to_fool, y_to_fool = x[ind_to_fool].clone(), y[ind_to_fool].clone()
            acc_curr, adv_curr = self.attack_single_run(x_to_fool, y_to_fool)
            ind_curr = (acc_curr == 0).nonzero().squeeze()
            #
            acc[ind_to_fool[ind_curr]] = 0
            adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
            if self.verbose:
                print('restart {} - robust accuracy: {:.2%} - cum. time: {:.1f} s'.format(
                    counter, acc.float().mean(), time.time() - startt))

        return acc, adv
