import torchbearer

from torchbearer.callbacks import Callback
import csv


class CSVLogger(Callback):
    """Callback to log metrics to a csv file.
    """

    def __init__(self, filename, separator=',', batch_granularity=False, write_header=True, append=False):
        """Construct a CSVLogger callback which logs metrics to the given file.

        :param filename: The name of the file to output to
        :type filename: str
        :param separator: The delimiter to use (e.g. comma, tab etc.)
        :type separator: str
        :param batch_granularity: If True, write on each batch, else on each epoch
        :type batch_granularity: bool
        :param write_header: If True, write the CSV header at the beginning of training
        :type write_header: bool
        :param append: If True, append to the file instead of replacing it
        :type append: bool
        """
        super().__init__()
        self.batch_granularity = batch_granularity
        self.filename = filename
        self.separator = separator
        if append:
            filemode = 'a+'
        else:
            filemode = 'w+'
        self.csvfile = open(self.filename, filemode, newline='')
        self.write_header = write_header

    def on_step_training(self, state):
        super().on_step_training(state)
        if self.batch_granularity:
            self._write_to_dict(state)

    def on_end_epoch(self, state):
        super().on_end_training(state)
        self._write_to_dict(state)

    def on_end(self, state):
        super().on_end(state)
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
