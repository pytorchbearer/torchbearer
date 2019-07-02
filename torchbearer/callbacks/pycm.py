from __future__ import print_function

import torch

import torchbearer
from torchbearer import Callback
from torchbearer.bases import cite
from torchbearer.metrics import EpochLambda, MetricList


pycm_bib = """
@article{Haghighi2018,
  doi = {10.21105/joss.00729},
  url = {https://doi.org/10.21105/joss.00729},
  year  = {2018},
  month = {may},
  publisher = {The Open Journal},
  volume = {3},
  number = {25},
  pages = {729},
  author = {Sepand Haghighi and Masoomeh Jasemi and Shaahin Hessabi and Alireza Zolanvari},
  title = {{PyCM}: Multiclass confusion matrix library in Python},
  journal = {Journal of Open Source Software}
}
"""


def _to_pyplot(normalize=False, title='Confusion matrix', cmap=None):
    """
    This function modified to plots the ConfusionMatrix object.
    Normalization can be applied by setting `normalize=True`.

    Code Reference :
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import itertools
    import numpy as np

    if cmap is None:
        cmap = plt.cm.Blues

    def handler(cm, state):
        string_state = {str(key): state[key] for key in state.keys()}
        plt_cm = []
        for i in cm.classes:
            row = []
            for j in cm.classes:
                row.append(cm.table[i][j])
            plt_cm.append(row)
        plt_cm = np.array(plt_cm)
        if normalize:
            plt_cm = plt_cm.astype('float') / plt_cm.sum(axis=1)[:, np.newaxis]
        plt.imshow(plt_cm, interpolation='nearest', cmap=cmap)
        plt.title(title.format(**string_state))
        plt.colorbar()
        tick_marks = np.arange(len(cm.classes))
        plt.xticks(tick_marks, cm.classes, rotation=45)
        plt.yticks(tick_marks, cm.classes)

        fmt = '.2f' if normalize else 'd'
        thresh = plt_cm.max() / 2.
        for i, j in itertools.product(range(plt_cm.shape[0]), range(plt_cm.shape[1])):
            plt.text(j, i, format(plt_cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if plt_cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('Actual')
        plt.xlabel('Predict')
        plt.show()
    return handler


@cite(pycm_bib)
class PyCM(Callback):
    """Create confusion matrices using the `PyCM library <https://github.com/sepandhaghighi/pycm>`_.

    Args:
        kwargs: Additional keyword args to pass to the `ConfusionMatrix` call
    """
    def __init__(self, **kwargs):
        import sys
        if sys.version_info[0] < 3:
            raise Exception('PyCM requires Python>=3!')
        self._handlers = []

        self.kwargs = kwargs

    def _add_metric(self, state):
        from pycm import ConfusionMatrix

        def make_cm(y_pred, y_true):
            _, y_pred = torch.max(y_pred, 1)
            cm = ConfusionMatrix(y_true.cpu().numpy(), y_pred.cpu().numpy(), **self.kwargs)
            for handler in self._handlers:
                handler(cm, state)

        my_metric = EpochLambda('pycm', make_cm, False)
        my_metric.reset(state)
        state[torchbearer.METRIC_LIST] = MetricList([state[torchbearer.METRIC_LIST], my_metric])

    def on_train(self):
        """Process this callback for training batches

        Returns:
            PyCM: self
        """
        _old_start_training = self.on_start_training

        def wrapper(state):
            _old_start_training(state)
            self._add_metric(state)

        self.on_start_training = wrapper
        return self

    def on_val(self):
        """Process this callback for validation batches

        Returns:
            PyCM: self
        """
        _old_start_validation = self.on_start_validation

        def wrapper(state):
            _old_start_validation(state)
            if state[torchbearer.DATA] is torchbearer.VALIDATION_DATA:
                self._add_metric(state)

        self.on_start_validation = wrapper
        return self

    def on_test(self):
        """Process this callback for test batches

        Returns:
            PyCM: self
        """
        _old_start_validation = self.on_start_validation

        def wrapper(state):
            _old_start_validation(state)
            if state[torchbearer.DATA] is torchbearer.TEST_DATA:
                self._add_metric(state)

        self.on_start_validation = wrapper
        return self

    def with_handler(self, handler):
        """Append the given output handler to the list of handlers

        Args:
            handler: A function of confusion and matrix and state

        Returns:
            PyCM: self
        """
        self._handlers.append(handler)
        return self

    def to_state(self, key):
        """Send `ConfusionMatrix` objects from this callback to the given state key

        Args:
            key (StateKey): The key to store the confusion matrix in

        Returns:
            PyCM: self
        """
        def handler(cm, state):
            state[key] = cm
        return self.with_handler(handler)

    def to_console(self):
        """Print `ConfusionMatrix` objects from this callback to the console

        Returns:
            PyCM: self
        """
        return self.with_handler(lambda cm, _: print(cm))

    def to_pycm_file(self, filename, address=True, overall_param=None, class_param=None, class_name=None):
        """Save `ConfusionMatrix` objects from this callback to `.pycm` files

        Args:
            filename (str): The name of the file, will be formatted with state to create unique filenames if desired

        See:
            `PyCM Source <https://github.com/sepandhaghighi/pycm/blob/master/pycm/pycm_obj.py>`_

        Returns:
            PyCM: self
        """
        def handler(cm, state):
            string_state = {str(key): state[key] for key in state.keys()}
            cm.save_stat(filename.format(**string_state), address=address, overall_param=overall_param,
                         class_param=class_param, class_name=class_name)
        return self.with_handler(handler)

    def to_html_file(self, filename, address=True, overall_param=None, class_param=None, class_name=None,
                     color=(0, 0, 0), normalize=False):
        """Save `ConfusionMatrix` objects from this callback to `.html` files

        Args:
            filename (str): The name of the file, will be formatted with state to create unique filenames if desired

        See:
            `PyCM Source <https://github.com/sepandhaghighi/pycm/blob/master/pycm/pycm_obj.py>`_

        Returns:
            PyCM: self
        """
        def handler(cm, state):
            string_state = {str(key): state[key] for key in state.keys()}
            cm.save_html(filename.format(**string_state), address=address, overall_param=overall_param,
                         class_param=class_param, class_name=class_name, color=color, normalize=normalize)
        return self.with_handler(handler)

    def to_csv_file(self, filename, address=True, overall_param=None, class_param=None, class_name=None,
                    matrix_save=True, normalize=False):
        """Save `ConfusionMatrix` objects from this callback to `.csv` files

        Args:
            filename (str): The name of the file, will be formatted with state to create unique filenames if desired

        See:
            `PyCM Source <https://github.com/sepandhaghighi/pycm/blob/master/pycm/pycm_obj.py>`_

        Returns:
            PyCM: self
        """
        def handler(cm, state):
            string_state = {str(key): state[key] for key in state.keys()}
            cm.save_csv(filename.format(**string_state), address=address, overall_param=overall_param,
                        class_param=class_param, class_name=class_name, matrix_save=matrix_save, normalize=normalize)
        return self.with_handler(handler)

    def to_obj_file(self, filename, address=True, save_stat=False, save_vector=True):
        """Save `ConfusionMatrix` objects from this callback to `.obj` files

        Args:
            filename (str): The name of the file, will be formatted with state to create unique filenames if desired

        See:
            `PyCM Source <https://github.com/sepandhaghighi/pycm/blob/master/pycm/pycm_obj.py>`_

        Returns:
            PyCM: self
        """
        def handler(cm, state):
            string_state = {str(key): state[key] for key in state.keys()}
            cm.save_obj(filename.format(**string_state), address=address, save_stat=save_stat, save_vector=save_vector)
        return self.with_handler(handler)

    def to_pyplot(self, normalize=False, title='Confusion matrix', cmap=None):
        """Plot `ConfusionMatrix` objects from this callback with `matplotlib.pyplot`

        Args:
            normalize (bool): If true, normalize the confusion matrix
            title (str): Title to use for the plot, will be formatted with state to create unique titles if desired
            cmap: Colour map object to use for the plot, defaults to `plt.cm.Blues` if None

        See:
            `PyCM Source <https://github.com/sepandhaghighi/pycm/blob/master/pycm/pycm_obj.py>`_

        Returns:
            PyCM: self
        """
        return self.with_handler(_to_pyplot(normalize=normalize, title=title, cmap=cmap))
