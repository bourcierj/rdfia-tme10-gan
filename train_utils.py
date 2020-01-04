import torch
from pathlib import Path


class Checkpoint():
    """A model checkpoint state."""
    def __init__(self, path='./checkpt.pt', **kwargs):

        self.path = Path(path)
        self.kwattrs = kwargs  # attributes passed as keyword-arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

    def state_dict(self):
        """Returns a dictionary containing the whole state of the checkpoint."""
        state = dict()
        for attr, value in self.kwattrs.items():
            try:
                state[attr] = getattr(self, attr, value).state_dict()
            except AttributeError:
                state[attr] = getattr(self, attr, value)
        return state

    def save(self, suffix=''):
        """Serializes the checkpoint.
        Args:
            suffix (str): if provided, a suffix will be prepended before the extension
                of the object's savepath attribute.
        """
        if suffix:
            savepath = self.path.parent / Path(self.path.stem + suffix +
                                               self.path.suffix)
        else:
            savepath = self.path
        with savepath.open('wb') as fp:
            torch.save(self.state_dict(), fp)

    def load(self, map_location=None):
        """Deserializes and maps the checkpoint to the available device.
        Args:
            map_location: a :class:`torch.device` specifying how to remap storage
            locations
        """
        with self.path.open('rb') as fp:
            state_dict = torch.load(fp, map_location)
            self.load_state_dict(state_dict)

    def load_state_dict(self, state_dict):
        """Copies parameters and from :attr:`state_dict` into the attributes of this
        checkpoint. The keys of :attr:`state_dict` must exactly match the keys returned
        by this checkpoint's :meth:`~state_dict` function.
        Args:
            state_dict (dict): a dictionary containing parameters
        """
        state_dict = state_dict.copy()
        for attr in self.kwattrs:
            # this will raise a key error if the state dicts aren't compatible
            # To DELETE
            try:
                for (k, v) in getattr(self, attr).state_dict().items():
                    if isinstance(v, torch.Tensor):
                        # print('tensor found')
                        other_v = state_dict[attr][k]
                        # print(k)
                        if torch.allclose(v, other_v):
                            print("Checkpoint.load:Warning: elements '{}' are equal between loaded state dict and original")
            except AttributeError:
                if getattr(self, attr) == state_dict[attr]:
                    print("Checkpoint.load:Warning: elements '{}' are equal between loaded state dict and original")

            try:
                try:
                    getattr(self, attr).load_state_dict(state_dict.pop(attr))
                except AttributeError:  # attr is not a module
                    setattr(self, attr, state_dict.pop(attr))
            except KeyError as e:
                raise KeyError("Missing key in argument 'state_dict': {}"
                               .format(str(e)))
        if len(state_dict) != 0:
            raise ValueError("Unexpected key(s) in argument 'state_dict': {}"
                             .format(list(state_dict.keys())))


def get_hparams_dict(args, ignore_keys=set()):
    """Get hyperparameters with values in a dict."""
    return {key.replace('_', '-'): val for key, val in vars(args).items()
               if key not in ignore_keys}


def get_experiment_name(prefix, hparams):
    """Generate a string name for the experiment."""
    return prefix + '_'.join([f"{key}={val}" for key, val in hparams.items()])


class BestLossTracker():
    """
    Tracks the best loss and associated values (for example iteration, accuracy...)
    Args:
        minimize (bool, default: True): If False, greater is better.
    """
    def __init__(self, minimize=True):
        if minimize:
            self.best_loss = float('inf')
            self.op = min
        else:
            self.best_loss = -float('inf')
            self.op = max
        # a flag that tells if the last update led to an improvement
        self.better = False

    def update(self, loss, **kwargs):
        """Updates the current best loss if the provided loss is better.
        Args:
            loss (float): the loss to check
            kwargs (dict): associated values to keep track of, provided as
                keyword arguments.
        """
        if self.op(self.best_loss, loss) == loss:  # improvement
            self.best_loss = loss
            for key, value in kwargs.items():
                setattr(self, key, value)
            self.better = True
        else:
            self.better = False


def test_best_loss_tracker():

    tracker = BestLossTracker()
    loss = 0.8
    tracker.update(loss)
    assert(tracker.best_loss == 0.8 and tracker.better)
    loss = 0.6
    accuracy = 0.38
    epoch = 5
    tracker.update(loss, accuracy=accuracy, epoch=epoch)
    assert(tracker.best_loss == 0.6 and tracker.better)
    assert(all(hasattr(tracker, attr) for attr in ('accuracy', 'epoch')))
    loss = 0.9
    ignore = 3.14151
    tracker.update(loss, ignore=ignore)
    assert(tracker.best_loss == 0.6 and not tracker.better)
    assert(not hasattr(tracker, 'ignore'))

def test_checkpoint_state():

    import os
    import glob
    import torch.nn as nn
    import torch.optim as optim

    module = nn.Linear(2, 2)
    optimizer = optim.SGD(module.parameters(), lr=0.01)
    checkpoint = Checkpoint(path='./checkpt_test.pt', module=module,
                                 optimizer=optimizer, epoch=10)
    checkpoint.save()
    checkpoint.save(suffix='_best')
    module = nn.Linear(2, 2)
    print('checkpoint state dict:')
    print(checkpoint.state_dict(), '\n')
    optimizer = optim.SGD(module.parameters(), lr=0.02)
    checkpoint2 = Checkpoint(path='./checkpt_test_best.pt', module=module,
                                  optimizer=optimizer, epoch=0)
    print('checkpoint2 state dict (before load):')
    print(checkpoint2.state_dict(), '\n')

    checkpoint2.load()
    print('checkpoint2 state dict (after load):')
    print(checkpoint2.state_dict(), '\n')
    for fp in glob.glob('./checkpt_test*.pt'):
        os.remove(fp)


if __name__ == '__main__':

    test_checkpoint_state()
