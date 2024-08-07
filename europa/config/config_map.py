# We have to add this ~type: method~ thing here because 
# transformers loads the scheduler and optimizer from a method,
# and torch/timm loads the optimizer from a class (that we can use load_obj) 
SCHEDULER_MAP = {
    "cosine-with-warmup": "get_cosine_schedule_with_warmup",
    "linear-with-warmup": "get_linear_schedule_with_warmup",
}

OPTIMIZER_MAP = {
    "adamw": "torch.optim.AdamW",
    "adam": "torch.optim.Adam",
    "sgd": "torch.optim.SGD",
    "rmsprop": "torch.optim.RMSprop",
}