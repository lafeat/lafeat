import math
import time

import torch

from .attack import LafeatAttack
from .targeted import TargetedLafeatAttack


class LafeatEval():
    def __init__(
            self, model, x, y, n_iter, norm, eps, betas=(0.0, ),
            target=False, batch_size=125, device='cuda', verbose=True):
        self.model = model
        self.x = x
        self.y = y
        self.norm = norm
        if norm not in ['Linf', 'L2']:
            raise ValueError(f'Unexpected norm {norm}.')
        self.epsilon = eps
        self.betas = betas
        rho = 0.75
        self.attack = LafeatAttack(
            self.model, n_iter, self.norm, self.epsilon,
            eot_iter=1, rho=rho, verbose=False, device=device)
        self.targeted_attack = TargetedLafeatAttack(
            self.model, n_iter, self.norm, self.epsilon,
            eot_iter=1, rho=rho, verbose=False, device=device)
        self.attacks = [self.attack]
        if target:
            self.attacks.append(self.targeted_attack)
        self.batch_size = batch_size
        self.verbose = verbose
        self.device = device
        self.robust = torch.ones(x.size(0), dtype=torch.bool).to(x.device)
        self.logits_first_minus_second_value = torch.ones(x.size(0), dtype=torch.float).to(x.device)
        self.x_adv = x.clone().detach()

    def _logits(self, x):
        return self.model(x)[-1]

    def _eval_batch(self, x, y):
        return y == self._logits(x).argmax(1)

    def _accuracy(self):
        return self.robust.sum()*1.0 / self.x.size(0)

    def _enumerate_batch(self):
        num_robust = int(self.robust.sum())
        if num_robust == 0:
            return
        num_batches = int(math.ceil(num_robust / self.batch_size))
        robust_index = self.robust.nonzero(as_tuple=False).squeeze()
        ########################################################
        before_sorted_robust_lin_idcs = torch.nonzero(self.robust, as_tuple=False)
        first_minus_second_value_robust = self.logits_first_minus_second_value[before_sorted_robust_lin_idcs]

        sorted_first_minus_second_value, indices_first_minus_second_value = torch.sort(first_minus_second_value_robust, dim=0)
        sorted_robust_lin_idcs = before_sorted_robust_lin_idcs[indices_first_minus_second_value]
        if num_robust > 1:
            sorted_robust_lin_idcs.squeeze_()
        for i in range(num_batches):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, num_robust)
            batch_index = sorted_robust_lin_idcs[start:end]
            xi = self.x[batch_index, :].clone().to(self.device)
            yi = self.y[batch_index].clone().to(self.device)
            yield batch_index, (xi, yi)

    def _init_pass(self):
        robust = []
        first_minus_second_value = []
        num_robust = int(self.robust.sum())
        if num_robust == 0:
            return
        num_batches = int(math.ceil(num_robust / self.batch_size))
        robust_index = self.robust.nonzero(as_tuple=False).squeeze()
        for i in range(num_batches):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, num_robust)
            batch_index = robust_index[start:end]
            xi = self.x[batch_index, :].clone().to(self.device)
            yi = self.y[batch_index].clone().to(self.device)
            correct = self._eval_batch(xi, yi)
            robust.append(correct.detach().bool())
            output = self._logits(xi)
            maxk = max((10,))
            pred_val_out, pred_id_out = output.topk(maxk, 1, True, True)
            first_minus_second_value_batch = (pred_val_out[:, 0] - pred_val_out[:, 1]).detach()
            first_minus_second_value.append(first_minus_second_value_batch)
        # before sort robustness
        self.robust = torch.cat(robust)
        self.logits_first_minus_second_value = torch.cat(first_minus_second_value)

        #################################################################################
        if self.verbose:
            print(f'Initial accuracy: {self._accuracy():.2%}.')


    def _attack_batch(self, attack, index, x, y, beta=None):
        # make sure that x is a 4d tensor
        # even if there is only a single datapoint left
        if len(x.shape) == 3:
            x.unsqueeze_(dim=0)
        # run attack
        _, adv = attack.perturb(x, y, scale_17=beta)
        adv_batch = ~self._eval_batch(adv, y)
        adv_index = index[adv_batch]
        return adv_index, adv[adv_batch].detach()

    def _attacks_to_run(self):
        for a in self.attacks:
            for b in self.betas:
                yield a, b

    def _single_attack(self, attack, beta, num_attack):
        num_batches = int(math.ceil(int(self.robust.sum()) / self.batch_size))
        num_robust = int(self.robust.sum())
        if num_robust > 1:
            self.robust.squeeze_()
        for b, (i, (x, y)) in enumerate(self._enumerate_batch()):
            adv_index, adv = self._attack_batch(attack, i, x, y, beta)

            self.robust[adv_index] = False
            self.x_adv[adv_index] = adv.to(self.x_adv.device)
            if not self.verbose:
                continue
            print(
                f'{attack.__class__.__name__}, beta: {beta:.3f}, '
                f'acc: {self._accuracy():.2%}, '
                f'batch: {b + 1}/{num_batches}, '
                f'perturbed: {adv.size(0) / x.size(0):.2%} '
                f'({adv.size(0)}/{x.size(0)}).')

    def _boundary_check(self):
        x = self.x
        x_adv = self.x_adv
        if self.norm == 'Linf':
            res = (x_adv - x).abs().max()
        elif self.norm == 'L2':
            res = ((x_adv - x) ** 2).view(x.size(0), -1).sum(-1).sqrt()
        else:
            raise NotImplementedError
        print(
            f'Max {self.norm} perturbation: {res.max():.5f}, '
            f'#NaN: {torch.isnan(x_adv).sum()}, '
            f'max: {x_adv.max():.5f}, min: {x_adv.min():.5f}.')

    def eval(self):
        if self.verbose:
            attacks = '\n  '.join(
                f'attack: {a.__class__.__name__}, beta: {b:.3f}'
                for a, b in self._attacks_to_run())
            print(f'Attacks to run:\n  {attacks}')
        with torch.no_grad():
            self._init_pass()
            time_start = time.time()
            num_attack = 0
            for (attack, beta) in self._attacks_to_run():
                num_attack += 1
                self._single_attack(attack, beta, num_attack)
                if not self.verbose:
                    continue
                print(
                    f'Robust accuracy after {attack.__class__.__name__}: '
                    f'{self._accuracy():.2%} '
                    f'(total time {time.time() - time_start:.1f} s)')
            if self.verbose:
                self._boundary_check()
                print(f'Final accuracy: {self._accuracy():.2%}')
        return self.x_adv
