from bink.callbacks import Callback
import csv


class CSVLogger(Callback):
    def __init__(self, filename, separator=',', batch_granularity=False, write_header=True, append=False):
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

    def on_end_training(self, state):
        super().on_end_training(state)
        self._write_to_dict(state)

    def on_end(self, state):
        super().on_end(state)
        self.csvfile.close()

    def _write_to_dict(self, state):
        fields = self._get_field_dict(state)

        if self.write_header:
            self.writer = csv.DictWriter(self.csvfile, fieldnames=fields.keys(), delimiter=self.separator)
            self.writer.writeheader()
            self.write_header = False

        self.writer.writerow(fields)
        self.csvfile.flush()

    def _get_field_dict(self, state):
        fields = {'epoch': state['epoch']}

        if self.batch_granularity:
            fields.update({'batch': state['t']})

        fields.update(state['metrics'])

        return fields
