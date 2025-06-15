import torch

from .learning_rate_schedule import learning_rate_schedule


class AdamW(torch.optim.Optimizer):

    def __init__(
        self,
        params: torch.Tensor,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        lr_max: float | None = None,
        lr_min: float | None = None,
        t_w: int | None = None,
        t_c: int | None = None,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.t_w = t_w
        self.t_c = t_c

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                # State initialization
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                m, v = state["m"], state["v"]
                state["step"] += 1
                t = state["step"]

                if self.lr_max is not None and self.lr_min is not None and \
                    self.t_w is not None and self.t_c is not None:
                    lr = learning_rate_schedule(
                        t, 
                        lr_max=self.lr_max,
                        lr_min=self.lr_min,
                        t_w=self.t_w,
                        t_c=self.t_c
                    )

                # Update the first moment estimate
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Update the second moment estimate
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                # Compute adjusted α for iteration t
                step_size = lr * ((1 - beta2 ** t) ** 0.5) / (1 - beta1 ** t)
                # Update the parameters
                p.data.addcdiv_(m, torch.sqrt(v) + eps, value=-step_size)
                # Apply weight decay
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)
