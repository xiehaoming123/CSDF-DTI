import datetime
from pathlib import Path

import torch


class EarlyStopping:
    def __init__(self, mode="higher", patience=10, filename=None, metric=None):
        if metric is not None:
            metric_modes = {
                "r2": "higher",
                "roc_auc_score": "higher",
                "pr_auc_score": "higher",
                "mae": "lower",
                "rmse": "lower",
            }
            if metric not in metric_modes:
                raise ValueError(f"Unsupported metric: {metric}")
            mode = metric_modes[metric]

        if mode not in {"higher", "lower"}:
            raise ValueError(f"Unsupported mode: {mode}")

        if filename is None:
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"early_stop_{stamp}.pth"

        self.mode = mode
        self.patience = int(patience)
        self.filename = str(filename)
        self.counter = 0
        self.timestep = 0
        self.best_score = None
        self.early_stop = False

    def _improved(self, score):
        if self.best_score is None:
            return True
        if self.mode == "higher":
            return score > self.best_score
        return score < self.best_score

    def step(self, score, model):
        self.timestep += 1
        score = float(score)

        if self._improved(score):
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def save_checkpoint(self, model):
        path = Path(self.filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "timestep": self.timestep,
                "best_score": self.best_score,
            },
            path,
        )

    def load_checkpoint(self, model, map_location=None, strict=True):
        state = torch.load(self.filename, map_location=map_location)
        model.load_state_dict(state["model_state_dict"], strict=strict)
        return model
