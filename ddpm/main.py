from ddpm import DDPM
from scheduler import Scheduler
from dataloader import DataManipulator
import argparse

if __name__ == '__main__':

    betas = Scheduler().linear_beta_scheduler()
    data = DataManipulator().get_data_loader()
    diffusion_model = DDPM(betas)

    diffusion_model.train(data)


