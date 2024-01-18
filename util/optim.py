import torch
import torch.optim as optim

class Optim:
    def __init__(self, optimizer: str, params, lr=0.05, lr_decay=1, momentum=None, weight_decay=None, start_decay_at=float('inf'), decay_factor=0.1, patience=5):
        self.optimizer = optimizer
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.decay_factor = decay_factor
        self.patience = patience
        self.num_bad_epochs = 0
        self.best_val_loss = float('inf')

        self.set_optimizer(params)

    def set_optimizer(self, params):
        optimizer_dict = {
            'SGD': optim.SGD,
            'Adadelta': optim.Adadelta,
            'Adam': optim.Adam,
            'Adamax': optim.Adamax,
            'AdamW': optim.AdamW,
            'RMSprop': optim.RMSprop,
            'Rprop': optim.Rprop
        }

        if self.optimizer in optimizer_dict:
            optimizer_class = optimizer_dict[self.optimizer]
            self.optimizer = optimizer_class(self.params, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Optimizer '{self.optimizer}' not supported.")

    def update_lr(self, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience and self.lr > self.start_decay_at:
            self.lr *= self.decay_factor
            self.num_bad_epochs = 0
            print(f"Learning rate decayed to {self.lr}")

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
