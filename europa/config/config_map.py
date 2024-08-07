SCHEDULER_MAP = {
    "steplr": "timm.scheduler.StepLRScheduler",
    "cosine": "timm.scheduler.CosineLRScheduler",
}

OPTIMIZER_MAP = {
    "adamw": "torch.optim.AdamW",
    "adam": "torch.optim.Adam",
    "sgd": "torch.optim.SGD",
    "rmsprop": "torch.optim.RMSprop",
}