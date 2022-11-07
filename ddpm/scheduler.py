import torch
import torch.nn.functional as F
from typing import Tuple


class Scheduler:
    def __init__(self, timesteps: int, beta: Tuple[float, float] ) -> None:
        self.timesteps = timesteps
        self.beta_start = beta[0]
        self.beta_end = beta[1]
        

    def linear_beta_scheduler(self) -> torch.Tensor
        return torch.linspace(self.beta_start, self.beta_end, self.timesteps)
