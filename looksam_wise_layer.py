import torch
import torch.nn as nn
from typing import Any, Callable

class LookSAM(torch.optim.Optimizer):

    def __init__(self,
                 k: int,
                 alpha: float,
                 model: nn.Module,
                 base_optimizer: torch.optim.Optimizer,
                 criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 rho: float = 0.05,
                 **kwargs: Any
    ):

        """
        LookSAM algorithm: https://arxiv.org/pdf/2203.02714.pdf
        Optimization algorithm that capable of simultaneously minimizing loss and loss sharpness to narrow
        the generalization gap.

        :param k: frequency of SAM's gradient calculation (default: 10)
        :param model: your network
        :param criterion: your loss function
        :param base_optimizer: optimizer module (SGD, Adam, etc...)
        :param alpha: scaling factor for the adaptive ratio (default: 0.7)
        :param rho: radius of the l_p ball (default: 0.1)

        :return: None

        Usage:
            model = YourModel()
            criterion = YourCriterion()
            base_optimizer = YourBaseOptimizer
            optimizer = LookSAM(k=k,
                                alpha=alpha,
                                model=model,
                                base_optimizer=base_optimizer,
                                criterion=criterion,
                                rho=rho,
                                **kwargs)

            ...

            for train_index, (samples, targets) in enumerate(loader):
                loss = criterion(model(samples), targets)
                loss.backward()
                optimizer.step(t=train_index, samples=samples, targets=targets, zero_sam_grad=True, zero_grad=True)

            ...

        """

        defaults = dict(alpha=alpha, rho=rho, **kwargs)
        self.model = model
        super(LookSAM, self).__init__(self.model.parameters(), defaults)

        self.k = k
        self.alpha = torch.tensor(alpha, requires_grad=False)
        self.criterion = criterion

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.criterion = criterion

    @staticmethod
    def normalized(g):
        return g / (g.norm(p=2) + 1e-8) # 必须在除数后加上一个补偿值1e-8，否则会出现loss=Nan的现象

    def step(self, t, samples, targets, zero_sam_grad=True, zero_grad=True,distributed=True):
        if not t % self.k:
            with torch.no_grad():
                group = self.param_groups[0]
                scale = group['rho'] / (self._grad_norm() + 1e-8)

            for index_p, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                self.state[p]['old_p'] = p.data.clone()
                self.state[f'old_grad_p_{index_p}']['old_grad_p'] = p.grad.clone()

                with torch.no_grad():
                    e_w = p.grad * scale.to(p)
                    p.add_(e_w)

            if zero_sam_grad:
                self.zero_grad()

            loss = self.criterion(self.model(samples), targets)
            loss.backward()

        group = self.param_groups[0]
        gv_bucket = []

        if t < self.k:
            for index_p, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                self.state[f'gv_{index_p}']['gv'] = torch.zeros(p.grad.shape,dtype=p.grad.dtype, device=p.grad.device)

        elif not t % self.k:

            for index_p, p in enumerate(group['params']):
                if p.grad is None:
                    continue


                with torch.no_grad():
                    old_grad_p = self.state[f'old_grad_p_{index_p}']['old_grad_p']
                    self.state[f'gv_{index_p}']['gv'] = torch.sum(torch.mul(p.grad,old_grad_p)) / (torch.sum(torch.pow(old_grad_p,2)) + 1e-8)
                    self.state[f'gv_{index_p}']['gv'] = old_grad_p * self.state[f'gv_{index_p}']['gv']
                    self.state[f'gv_{index_p}']['gv'] = torch.sub(p.grad,self.state[f'gv_{index_p}']['gv'])

                    gv_bucket.append(self.state[f'gv_{index_p}']['gv'])
                p.data = self.state[p]['old_p']

        else:

            for index_p, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                with torch.no_grad():
                    gv_ = self.state[f'gv_{index_p}']['gv']
                    g_norm = p.grad.norm(p=2)
                    g_v_norm = gv_.norm(p=2)
                    if g_v_norm == 0.:
                        norm = 1.
                    else:
                        norm = g_norm / (g_v_norm + 1e-8)

                    norm = min(norm, 1.)
                    p.grad.add_(self.alpha.to(p) * norm * gv_)


        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]['params'][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device) for group in self.param_groups for p in group['params']
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def _gv_norm(self):
        shared_device = self.param_groups[0]['params'][0].device
        norm = torch.norm(
            torch.stack([
                self.state[f'gv_{i}']['gv'].norm(p=2).to(shared_device) for i in range(len(self.param_groups[0]['params']))
                if self.state[f'gv_{i}']['gv'] is not None
            ]),
            p=2
        )
        return norm



