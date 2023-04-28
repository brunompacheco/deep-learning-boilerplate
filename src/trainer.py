import logging
import random
from abc import ABC, abstractmethod
from contextlib import nullcontext
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader

from src.utils import timeit


class Trainer(ABC):
    """Generic trainer for PyTorch NNs.

    Attributes:
        net: the neural network to be trained.
        epochs: number of epochs to train the network.
        lr: learning rate.
        optimizer: optimizer (name of a optimizer inside `torch.optim`).
        loss_func: a valid PyTorch loss function.
        lr_scheduler: if a scheduler is to be used, provide the name of a valid
        `torch.optim.lr_scheduler`.
        lr_scheduler_params: parameters of selected `lr_scheduler`.
        device: see `torch.device`.
        wandb_project: W&B project where to log and store model.
        logger: see `logging`.
        random_seed: if not None (default = 42), fixes randomness for Python,
        NumPy as PyTorch (makes trainig reproducible).
    """
    def __init__(self, net: nn.Module, dataset: Dataset, epochs=5, lr= 0.1,
                 batch_size=2**4, optimizer: str = 'Adam',
                 optimizer_params=dict(), loss_func: str = 'MSELoss',
                 loss_func_params=dict(), lr_scheduler: str = None,
                 lr_scheduler_params=dict(), mixed_precision=True,
                 device=None, wandb_project=None, wandb_group=None, logger=None,
                 checkpoint_every=50, random_seed=42, max_loss=None) -> None:
        self._is_initalized = False

        if device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        self._e = 0  # inital epoch

        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

        self.net = net.to(self.device)
        self.dataset = dataset
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.loss_func = loss_func
        self.loss_func_params = loss_func_params
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params

        self._dtype = next(self.net.parameters()).dtype

        self.mixed_precision = mixed_precision

        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.l = logging.getLogger(__name__)
        else:
            self.l = logger

        self.checkpoint_every = checkpoint_every

        self.random_seed = random_seed
        if self.random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
            torch.manual_seed(random_seed)

        self.best_val = float('inf')

        self._log_to_wandb = False if wandb_project is None else True
        self.wandb_project = wandb_project
        self.wandb_group = wandb_group

        self.max_loss = max_loss

    def _load_optim(self, state_dict=None):
        Optimizer = eval(f"torch.optim.{self.optimizer}")
        self._optim = Optimizer(
            filter(lambda p: p.requires_grad, self.net.parameters()),
            lr=self.lr,
            **self.optimizer_params
        )
        if state_dict is not None:
            self._optim.load_state_dict(state_dict)
    
    def _load_lr_scheduler(self, state_dict=None):
        Scheduler = eval(f"torch.optim.lr_scheduler.{self.lr_scheduler}")
        self._scheduler = Scheduler(self._optim, **self.lr_scheduler_params)

        if state_dict is not None:
            self._scheduler.load_state_dict(state_dict)

    def _load_loss_func(self):
        LossClass = eval(f"nn.{self.loss_func}")
        self._loss_func = LossClass(**self.loss_func_params)

    @classmethod
    def load_trainer(cls, net: nn.Module, run_id: str, wandb_project=None,
                     logger=None):
        """Load a previously initialized trainer from wandb.

        Loads checkpoint from wandb and creates the instance.

        Args:
            run_id: valid wandb id.
            logger: same as the attribute.
        """
        wandb.init(
            project=wandb_project,
            entity="brunompac",
            id=run_id,
            resume='must',
        )

        # load checkpoint file
        checkpoint_file = wandb.restore('checkpoint.tar')
        checkpoint = torch.load(checkpoint_file.name)

        # load model
        net = net.to(wandb.config['device'])
        net.load_state_dict(checkpoint['model_state_dict'])

        # fix for older versions
        if 'lr_scheduler' not in wandb.config.keys():
            wandb.config['lr_scheduler'] = None
            wandb.config['lr_scheduler_params'] = None

        # create trainer instance
        self = cls(
            epochs=wandb.config['epochs'],
            net=net,
            lr=wandb.config['learning_rate'],
            optimizer=wandb.config['optimizer'],
            loss_func=wandb.config['loss_func'],
            lr_scheduler=wandb.config['lr_scheduler'],
            lr_scheduler_params=wandb.config['lr_scheduler_params'],
            device=wandb.config['device'],
            logger=logger,
            wandb_project=wandb_project,
            random_seed=wandb.config['random_seed'],
        )

        if 'best_val' in checkpoint.keys():
            self.best_val = checkpoint['best_val']

        self._e = checkpoint['epoch'] + 1

        self.l.info(f'Resuming training of {wandb.run.name} at epoch {self._e}')

        # load optimizer
        self._load_optim(checkpoint['optimizer_state_dict'])

        # load scheduler
        if self.lr_scheduler is not None:
            self._load_lr_scheduler(state_dict=checkpoint['scheduler_state_dict'])

        self._load_loss_func(self)

        if self.mixed_precision:
            self._scaler = GradScaler()
            self.autocast_if_mp = autocast
        else:
            self.autocast_if_mp = nullcontext

        self.train_data, self.val_data = self.prepare_data()

        self._is_initalized = True

        return self

    def setup_training(self):
        self.l.info('Setting up training')

        self._load_optim(state_dict=None)

        if self.lr_scheduler is not None:
            self._load_lr_scheduler(state_dict=None)

        if self._log_to_wandb:
            self._add_to_wandb_config({
                "learning_rate": self.lr,
                "epochs": self.epochs,
                "model": type(self.net).__name__,
                "optimizer": self.optimizer,
                "lr_scheduler": self.lr_scheduler,
                "lr_scheduler_params": self.lr_scheduler_params,
                "mixed_precision": self.mixed_precision,
                "loss_func": self.loss_func,
                "random_seed": self.random_seed,
                "device": self.device,
            })

            self.l.info('Initializing wandb.')
            self.initialize_wandb()

        self._load_loss_func()

        if self.mixed_precision:
            self._scaler = GradScaler()
            self.autocast_if_mp = autocast
        else:
            self.autocast_if_mp = nullcontext

        self.l.info('Preparing data')
        self.train_data, self.val_data = self.prepare_data()

        self._is_initalized = True

    def _add_to_wandb_config(self, d: dict):
        if not hasattr(self, '_wandb_config'):
            self._wandb_config = dict()

        for k, v in d.items():
            self._wandb_config[k] = v

    def initialize_wandb(self):
        wandb.init(
            project=self.wandb_project,
            entity="brunompac",
            group=self.wandb_group,
            config=self._wandb_config,
        )

        wandb.watch(self.net)

        self._id = wandb.run.id

        self.l.info(f"Wandb set up. Run ID: {self._id}")

    def prepare_data(self):
        """Splits dataset into training and validation data. Instantiate loaders.
        """
        generator = torch.Generator().manual_seed(33)

        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        train_data, val_data = torch.utils.data.random_split(
            self.dataset,
            (train_size, val_size),
            generator=generator
        )

        train_loader = DataLoader(train_data, self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, self.batch_size, shuffle=False)

        return train_loader, val_loader

    @staticmethod
    def _add_data_to_log(data: dict, prefix: str, data_to_log=dict()):
        for k, v in data.items():
            if k != 'all':
                data_to_log[prefix+k] = v
        
        return data_to_log

    def _run_epoch(self):
        # train
        train_time, (train_losses, train_times) = timeit(self.train_pass)()

        self.l.info(f"Training pass took {train_time:.3f} seconds")
        self.l.info(f"Training loss = {train_losses['all']}")

        # validation
        val_time, (val_losses, val_times) = timeit(self.validation_pass)()

        self.l.info(f"Validation pass took {val_time:.3f} seconds")
        self.l.info(f"Validation loss = {val_losses['all']}")

        data_to_log = {
            "train_loss": train_losses['all'],
            "val_loss": val_losses['all'],
            "train_time": train_time,
            "val_time": val_time,
        }
        self._add_data_to_log(train_losses, 'train_loss_', data_to_log)
        self._add_data_to_log(val_losses, 'val_loss_', data_to_log)
        self._add_data_to_log(train_times, 'train_time_', data_to_log)
        self._add_data_to_log(val_times, 'val_time_', data_to_log)

        val_score = val_losses['all']  # defines best model

        return data_to_log, val_score

    def run(self) -> nn.Module:
        if not self._is_initalized:
            self.setup_training()

        while self._e < self.epochs:
            self.l.info(f"Epoch {self._e} started ({self._e+1}/{self.epochs})")
            epoch_start_time = time()

            data_to_log, val_score = self._run_epoch()

            if self._log_to_wandb:
                wandb.log(data_to_log, step=self._e, commit=True)

                if self._e % self.checkpoint_every == self.checkpoint_every - 1:
                    self.l.info(f"Saving checkpoint")
                    self.save_checkpoint()

            if val_score < self.best_val:
                if self._log_to_wandb:
                    self.l.info(f"Saving best model")
                    self.save_model(name='model_best')

                self.best_val = val_score

            epoch_end_time = time()
            self.l.info(
                f"Epoch {self._e} finished and took "
                f"{epoch_end_time - epoch_start_time:.2f} seconds"
            )

            if self.max_loss is not None:
                if val_score > self.max_loss:
                    break

            self._e += 1

        if self._log_to_wandb:
            self.l.info(f"Saving model")
            self.save_model(name='model_last')

            wandb.finish()

        self.l.info('Training finished!')

        return self.net

    def optim_step(self, loss):
        if self.mixed_precision:
            backward_time, _  = timeit(self._scaler.scale(loss).backward)()
            self._scaler.step(self._optim)
            self._scaler.update()
        else:
            backward_time, _  = timeit(loss.backward)()
            self._optim.step()

        self._optim.zero_grad()
        return backward_time
    
    def get_loss_and_metrics(self, y_hat, y, validation=False):
        loss_time, loss =  timeit(self._loss_func)(y_hat.view_as(y), y)

        metrics = None
        if validation:
            # here you can compute performance metrics (populate `metrics`)
            pass

        return loss_time, loss, metrics
    
    def aggregate_loss_and_metrics(self, loss, size, metrics=None):
        # scale to data size
        loss = loss / size

        losses = {
            'all': loss
            # here you can aggregate metrics computed on the validation set and
            # track them on wandb
        }

        return losses

    def data_pass(self, data, train=False):
        loss = 0
        size = 0
        metrics = list()

        forward_time = 0
        loss_time = 0
        backward_time = 0

        with torch.set_grad_enabled(train):
            for X, y in data:
                X = X.to(self.device)
                y = y.to(self.device)

                with self.autocast_if_mp():
                    batch_forward_time, y_hat = timeit(self.net)(X)
                    forward_time += batch_forward_time

                    batch_loss_time, batch_loss, batch_metrics = self.get_loss_and_metrics(y_hat, y, validation=not train)
                loss_time += batch_loss_time

                metrics.append(batch_metrics)

                if train:
                    batch_backward_time = self.optim_step(batch_loss)
                    backward_time += batch_backward_time

                loss += batch_loss.item() * len(y)
                size += len(y)

            if self.lr_scheduler is not None:
                self._scheduler.step()

        if all([b_m is None for b_m in metrics]):
            metrics = None

        losses = self.aggregate_loss_and_metrics(loss, size, metrics)
        times = {
            'forward': forward_time,
            'loss': loss_time,
            'backward': backward_time,
        }

        return losses, times

    def train_pass(self):
        self.net.train()
        return self.data_pass(self.train_data, train=True)

    def validation_pass(self):
        self.net.eval()
        return self.data_pass(self.val_data, train=False)

    def save_checkpoint(self):
        checkpoint = {
            'epoch': self._e,
            'best_val': self.best_val,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self._optim.state_dict(),
        }

        if self.lr_scheduler is not None:
            checkpoint['scheduler_state_dict'] = self._scheduler.state_dict()

        torch.save(checkpoint, Path(wandb.run.dir)/'checkpoint.tar')
        wandb.save('checkpoint.tar')

    def save_model(self, name='model'):
        fname = f"{name}.pth"
        fpath = Path(wandb.run.dir)/fname

        torch.save(self.net.state_dict(), fpath)
        wandb.save(fname)

        return fpath
