import math


def learning_rate_schedule(
    t: int,
    lr_max: float,
    lr_min: float,
    t_w: int,
    t_c: int,
) -> float:
    """
    Schedule learing rate

    Args:
        t (int): the current iteration
        lr_max (float): the maximum learning rate α_max
        lr_min (float): the minimum (final) learning rate α_min
        t_w (int): the number of warm-up iterations Tw
        t_c (int): the number of cosine annealing iterations

    Returns:
        float: current learing rate
    """
    # Warm-up
    if t < t_w:
        return t / t_w * lr_max
    # Post-annealing
    elif t > t_c:
        return lr_min
    # Cosine annealing
    else:
        return lr_min + (0.5*(1 + math.cos(
            ((t - t_w)/(t_c - t_w))*math.pi)
        )*(lr_max - lr_min))
