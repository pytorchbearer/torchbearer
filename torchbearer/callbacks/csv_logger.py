import sys

import torchbearer

from torchbearer.callbacks import Callback
import csv


class CSVLogger(Callback):
    """Callback to log metrics to a given csv file.

    Example: ::

        >>> from torchbearer.callbacks import CSVLogger
        >>> from torchbearer import Trial
        >>> import torch

        # Example Trial (without optimiser or loss criterion) which writes metrics to a csv file appending to previous content
        >>> logger = CSVLogger('my_path.pt', separator=',', append=True)
        >>> trial = Trial(None, callbacks=[logger], metrics=['acc'])

    Args:
        filename (str): The name of the file to output to
        separator (str): The delimiter to use (e.g. comma, tab etc.)
        batch_granularity (bool): If True, write on each batch, else on each epoch
        write_header (bool): If True, write the CSV header at the beginning of training
        append (bool): If True, append to the file instead of replacing it

    State Requirements:
        - :attr:`torchbearer.state.EPOCH`: State should have the current epoch stored
        - :attr:`torchbearer.state.METRICS`: Metrics dictionary should exist
        - :attr:`torchbearer.state.BATCH`: State should have the current batch stored if using `batch_granularity`
    """

    def __init__(self, filename, separator=',', batch_granularity=False, write_header=True, append=False):

        super(CSVLogger, self).__init__()
        self.batch_granularity = batch_granularity
        self.filename = filename
        self.separator = separator
        if append:
            filemode = 'a'
        else:
            filemode = 'w'

        if sys.version_info[0] < 3:
            filemode += 'b'
            self.csvfile = open(self.filename, filemode)
        else:
            self.csvfile = open(self.filename, filemode, newline='')
        self.write_header = write_header

    def on_step_training(self, state):
        super(CSVLogger, self).on_step_training(state)
        if self.batch_granularity:
            self._write_to_dict(state)

    def on_end_epoch(self, state):
        super(CSVLogger, self).on_end_training(state)
        self._write_to_dict(state)

    def on_end(self, state):
        super(CSVLogger, self).on_end(state)
        self.csvfile.close()

    def _write_to_dict(self, state):
        fields = self._get_field_dict(state)
        self.writer = csv.DictWriter(self.csvfile, fieldnames=fields.keys(), delimiter=self.separator)

        if self.write_header:
            self.writer.writeheader()
            self.write_header = False

        self.writer.writerow(fields)
        self.csvfile.flush()

    def _get_field_dict(self, state):
        fields = {'epoch': state[torchbearer.EPOCH]}

        if self.batch_granularity:
            fields.update({'batch': state[torchbearer.BATCH]})

        fields.update(state[torchbearer.METRICS])

        return fields
