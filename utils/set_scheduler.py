from torch.optim import lr_scheduler
from utils.custom_lr import CosineAnnealingWarmUpRestarts


def choose_scheduler(optimizer, scheduler_type, **kwargs):
    if scheduler_type == "StepLR":
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs["Train"]["step_size"],
            gamma=kwargs["Train"]["gamma"],
        )

    elif scheduler_type == "ExponentialLR":
        scheduler = lr_scheduler.ExponentialLR(
            optimizer, gamma=kwargs["Train"]["gamma"]
        )

    elif scheduler_type == "CyclicLR":
        scheduler = lr_scheduler.CyclicLR(
            optimizer,
            base_lr=kwargs["Train"]["base_lr"],
            max_lr=kwargs["Train"]["max_lr"],
            step_size_up=kwargs["Train"]["step_size_up"],
            step_size_down=kwargs.get(
                "step_size_down", kwargs["Train"]["step_size_up"]
            ),
            mode=kwargs["Train"]["mode"],
            gamma=kwargs["Train"]["gamma"],
        )

    elif scheduler_type == "CosineAnnealingWarmUp":
        scheduler = CosineAnnealingWarmUpRestarts(
            optimizer,
            T_0=kwargs["Train"]["T_0"],
            T_mult=kwargs["Train"]["T_mult"],
            eta_max=kwargs["Train"]["eta_max"],
            T_up=kwargs["Train"]["T_up"],
            gamma=kwargs["Train"]["gamma"],
        )
    else:
        print("Invalid scheduler_type")
        return None

    return scheduler
