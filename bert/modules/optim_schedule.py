'''A wrapper class for optimizer '''
import numpy as np


class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, n_warmup_steps, total_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.total_steps = total_steps
        self.n_current_steps = 0
        # self.init_lr = np.power(d_model, -0.5)
        self.init_lr = self._optimizer.param_groups[0]['lr']

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):  # get lr scale based on current step and warmup steps

        # return np.min([
        #     np.power(self.n_current_steps, -0.5),  # as the learning rate increases, the scale decreases.
        #     np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])  # as the warmup steps increase, the scale increases.

        if self.n_current_steps < self.n_warmup_steps:
            return float(self.n_current_steps) / float(self.n_warmup_steps)

        return max(0.0, 1.0 - float(self.n_current_steps - self.n_warmup_steps) / float(self.total_steps - self.n_warmup_steps))

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1  # update the number of steps
        lr = self.init_lr * self._get_lr_scale()  #

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
