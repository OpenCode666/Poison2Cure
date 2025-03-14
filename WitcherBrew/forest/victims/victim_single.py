"""Single model default victim class."""

import torch
import numpy as np
from collections import defaultdict


from ..utils import set_random_seed
from ..consts import BENCHMARK
torch.backends.cudnn.benchmark = BENCHMARK

from .victim_base import _VictimBase


class _VictimSingle(_VictimBase):
    """Implement model-specific code and behavior for a single model on a single GPU.

    This is the simplest victim implementation.

    """

    """ Methods to initialize a model."""

    def initialize(self, seed=None):
        if self.args.modelkey is None:
            if seed is None:
                self.model_init_seed = np.random.randint(0, 2**32 - 1)
            else:
                self.model_init_seed = seed
        else:
            self.model_init_seed = self.args.modelkey
        set_random_seed(self.model_init_seed)
        self.model_name = self.args.net[0]
        self.model, self.defs, self.criterion, self.optimizer, self.scheduler, self.weight_FT_PT = self._initialize_model(self.args.net[0])

        self.model.to(**self.setup)
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        print(f'{self.args.net[0]} model initialized with random key {self.model_init_seed}.')

    def load_pretrained(self, model_dir):
        """Load a pretrained model."""
        import os
        
        # self.model.module.load_state_dict(torch.load(os.path.join(model_dir, self.model_name + '.pth')))   # multiple GPUs
        self.model.load_state_dict(torch.load(os.path.join(model_dir, self.model_name + '.pth'),map_location='cuda:0'))     # single GPU //'cuda:0'
        print(f'Loaded pretrained model from {os.path.join(model_dir, self.model_name )}')

    """ METHODS FOR (CLEAN) TRAINING AND TESTING OF BREWED POISONS"""

    def _iterate(self, kettle, poison_delta, max_epoch=None):
        """Validate a given poison by training the model and checking target accuracy."""
        stats = defaultdict(list)

        if max_epoch is None:
            max_epoch = self.defs.epochs

        def loss_fn(model, outputs, labels):
            return self.criterion(outputs, labels)

        single_setup = (self.model, self.defs, self.criterion, self.optimizer, self.scheduler)
        for self.epoch in range(max_epoch):
            print(f'Validating valid model at epoch {self.epoch} ...')
            self._step(kettle, poison_delta, loss_fn, self.epoch, stats, *single_setup)
            if self.args.dryrun:
                break
        return stats

    def step(self, kettle, poison_delta, poison_targets, true_classes):
        """Step through a model epoch. Optionally: minimize target loss."""
        stats = defaultdict(list)

        def loss_fn(model, outputs, labels):
            normal_loss = self.criterion(outputs, labels)
            model.eval()
            if self.args.adversarial != 0:
                target_loss = 1 / self.defs.batch_size * self.criterion(model(poison_targets), true_classes)
            else:
                target_loss = 0
            model.train()
            return normal_loss + self.args.adversarial * target_loss

        single_setup = (self.model, self.criterion, self.optimizer, self.scheduler)
        self._step(kettle, poison_delta, loss_fn, self.epoch, stats, *single_setup)
        self.epoch += 1
        if self.epoch > self.defs.epochs:
            self.epoch = 0
            print('Model reset to epoch 0.')
            self.model, self.criterion, self.optimizer, self.scheduler = self._initialize_model()
            self.model.to(**self.setup)
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)
        return stats

    """ Various Utilities."""

    def eval(self, dropout=False):
        """Switch everything into evaluation mode."""
        def apply_dropout(m):
            if type(m) == torch.nn.Dropout:
                m.train()
        self.model.eval()
        if dropout:
            self.model.apply(apply_dropout)

    def reset_learning_rate(self):
        """Reset scheduler object to initial state."""
        _, _, self.optimizer, self.scheduler = self._initialize_model()

    # pure_test
    def grad_UE_target_loss_original(self, auth_weight, UE_auth_loader, UE_priv_loader):
        loss_auth = 0.
        loss_priv = 0.
        total_auth = 0
        total_priv = 0
        self.model.train()
        for data, label, indeces in UE_auth_loader:
            indeces = indeces.cpu().numpy().astype(np.int64).squeeze()
            data = data.to(**self.setup)
            label = label.to(**self.setup)
            output = self.model(data,indeces)
            loss_auth += self.criterion(output, label)
            total_auth += label.size(0)
        for data, _, indeces, intend_label in UE_priv_loader:
            indeces = indeces.cpu().numpy().astype(np.int64).squeeze()
            data = data.to(**self.setup)
            intend_label = intend_label.to(**self.setup)
            output = self.model(data,indeces)
            loss_priv += self.criterion(output, intend_label)
            total_priv += intend_label.size(0)
        if total_priv == 0:
            loss = auth_weight * loss_auth / total_auth
        else:
            loss = auth_weight * loss_auth / total_auth + loss_priv / total_priv
        print(f'loss_auth: {loss_auth}, loss_priv: {loss_priv}, total_loss: {loss}')
        gradients = torch.autograd.grad(loss, self.model.parameters(), only_inputs=True)
        grad_norm = 0
        for grad in gradients:
            grad_norm += grad.detach().pow(2).sum()
        grad_norm = grad_norm.sqrt()
        return gradients, grad_norm
    def grad_UE_target_loss(self, auth_weight, UE_auth_loader, UE_priv_loader):
        loss_auth = 0.
        loss_priv = 0.
        total_auth = 0
        total_priv = 0
        self.model.train()
        for data, label, indeces in UE_auth_loader:
            indeces = indeces.cpu().numpy().astype(np.int64).squeeze()
            data = data.to(**self.setup)
            label = label.to(**self.setup)
            output = self.model(data,indeces)
            loss_auth += self.criterion(output, label)
            total_auth += label.size(0)
        # for data, label, indeces, _ in UE_priv_loader:
        for data, label, indeces in UE_priv_loader:
            indeces = indeces.cpu().numpy().astype(np.int64).squeeze()
            data = data.to(**self.setup)
            label = label.to(**self.setup)
            output = self.model(data,indeces)
            loss_priv += self.criterion(output, label)
            total_priv += label.size(0)
        if total_priv == 0:
            loss = auth_weight * loss_auth / total_auth
        else:
            loss = auth_weight * loss_auth / total_auth - loss_priv / total_priv
            # loss = (auth_weight * loss_auth - loss_priv) / (total_auth + total_priv)
        print(f'loss_auth: {loss_auth}, loss_priv: {loss_priv}, total_loss: {loss}')
        gradients = torch.autograd.grad(loss, self.model.parameters(), only_inputs=True)
        grad_norm = 0
        for grad in gradients:
            grad_norm += grad.detach().pow(2).sum()
        grad_norm = grad_norm.sqrt()
        return gradients, grad_norm
    # FT
    def grad_UE_target_loss_FT(self, auth_weight, UE_auth_loader, UE_priv_loader):
        loss_auth = 0.
        loss_priv = 0.
        total_auth = 0
        total_priv = 0
        self.model.train()
        for data, label, indeces in UE_auth_loader:
            indeces = indeces.cpu().numpy().astype(np.int64).squeeze()
            data = data.to(**self.setup)
            label = label.to(**self.setup)
            output = self.model(data,indeces)
            loss_auth += self.criterion(output, label)
            total_auth += label.size(0)
        for data, label, indeces in UE_priv_loader:
            indeces = indeces.cpu().numpy().astype(np.int64).squeeze()
            data = data.to(**self.setup)
            label = label.to(**self.setup)
            output = self.model(data,indeces)
            loss_priv += self.criterion(output, label)
            total_priv += label.size(0)
        if total_priv == 0:
            loss = auth_weight * loss_auth / total_auth
        else:
            loss = auth_weight * loss_auth / total_auth - loss_priv / total_priv
            # loss = (auth_weight * loss_auth- loss_priv) / (total_auth  + total_priv)
        print(f'loss_auth: {loss_auth}, loss_priv: {loss_priv}, total_loss: {loss}')
        gradients = torch.autograd.grad(loss, self.model.parameters(), only_inputs=True)
        grad_norm = 0
        for grad in gradients:
            grad_norm += grad.detach().pow(2).sum()
        grad_norm = grad_norm.sqrt()
        return gradients, grad_norm

    def gradient(self, images, labels, criterion=None):
        """Compute the gradient of criterion(model) w.r.t to given data."""
        if criterion is None:
            loss = self.criterion(self.model(images), labels)
        else:
            loss = criterion(self.model(images), labels)
        gradients = torch.autograd.grad(loss, self.model.parameters(), only_inputs=True)
        grad_norm = 0
        for grad in gradients:
            grad_norm += grad.detach().pow(2).sum()
        grad_norm = grad_norm.sqrt()
        return gradients, grad_norm

    def compute(self, function, *args):
        r"""Compute function on the given optimization problem, defined by criterion \circ model.

        Function has arguments: model, criterion
        """
        return function(self.model, self.model_name, self.criterion, self.weight_FT_PT, *args)
