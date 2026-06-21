import pickle
from pathlib import Path

import torch


def save_model_dict(model, model_dir, name):
    path = Path(model_dir) / f"{name}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return str(path)


def load_model_dict(model, ckpt, map_location=None, strict=True):
    state = torch.load(ckpt, map_location=map_location)
    model.load_state_dict(state, strict=strict)
    return model


def write_pickle(filename, obj):
    with open(filename, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


class BestMeter:
    def __init__(self, mode):
        self.mode = mode
        self.count = 0
        self.reset()

    def reset(self):
        self.best = float("inf") if self.mode == "min" else -float("inf")
        self.count = 0

    def update(self, value):
        self.best = value
        self.count = 0

    def get_best(self):
        return self.best

    def counter(self):
        self.count += 1
        return self.count


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def get_average(self):
        return self.sum / max(self.count, 1)
