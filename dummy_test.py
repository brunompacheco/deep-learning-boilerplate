import torch
import torch.nn as nn
from torch.utils.data import Dataset
from src.trainer import Trainer
from src.utils import timeit


class DummyDataset(Dataset):
    def __init__(self, n=100) -> None:
        super().__init__()

        X_neg = torch.normal(torch.zeros(n, 2), 0.5 * torch.ones(n, 2))
        X_pos = torch.normal(torch.ones(n, 2), 0.5 * torch.ones(n, 2))

        y = torch.ones(2*n)
        y[:n] = 0

        self.X = torch.vstack((X_neg, X_pos))
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]

class DummyTrainer(Trainer):
    def __init__(self, net: nn.Module, dataset: Dataset, epochs=5, lr=0.01,
                 batch_size=2**4, optimizer: str = 'Adam',
                 optimizer_params=dict(), loss_func: str = 'BCEWithLogitsLoss',
                 loss_func_params=dict(), lr_scheduler: str = None,
                 lr_scheduler_params=dict(), mixed_precision=True,
                 device=None, wandb_project=None, wandb_group=None, logger=None,
                 checkpoint_every=50, random_seed=42, max_loss=None) -> None:
        super().__init__(net, dataset, epochs, lr, batch_size, optimizer,
                         optimizer_params, loss_func, loss_func_params, lr_scheduler,
                         lr_scheduler_params, mixed_precision, device,
                         wandb_project, wandb_group, logger, checkpoint_every,
                         random_seed, max_loss)

    def get_loss_and_metrics(self, y_hat, y, validation=False):
        y_hat = y_hat.view_as(y)
        loss_time, loss =  timeit(self._loss_func)(y_hat, y)

        metrics = None
        if validation:
            y_pred = (y_hat > 0.5).to(y_hat)
            metrics = (y == y_pred).sum()

        return loss_time, loss, metrics
    
    def aggregate_loss_and_metrics(self, loss, size, metrics=None):
        # scale to data size
        loss = loss / size

        losses = {
            'all': loss
            # here you can aggregate metrics computed on the validation set and
            # track them on wandb
        }

        if metrics is not None:
            losses['accuracy'] = sum(metrics) / size

        return losses
    
    def _run_epoch(self):
        # train
        train_time, (train_losses, train_times) = timeit(self.train_pass)()

        self.l.info(f"Training pass took {train_time:.3f} seconds")
        self.l.info(f"Training loss = {train_losses['all']}")

        # validation
        val_time, (val_losses, val_times) = timeit(self.validation_pass)()

        self.l.info(f"Validation pass took {val_time:.3f} seconds")
        self.l.info(f"Validation loss = {val_losses['all']}")
        self.l.info(f"Validation accuracy = {val_losses['accuracy']}")

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

if __name__ == '__main__':
    h = 5
    dummy_net = nn.Sequential(
        nn.Linear(2, h),
        nn.ReLU(),
        nn.Linear(h, h),
        nn.ReLU(),
        nn.Linear(h, 1)
    )
    DummyTrainer(
        dummy_net,
        DummyDataset(),
        epochs=10,
    ).run()