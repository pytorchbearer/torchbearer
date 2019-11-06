The Metric API
====================================

There are a few levels of complexity to the metric API. You've probably already seen keys such as 'acc' and 'loss' can
be used to reference pre-built metrics, so we'll have a look at how these get mapped 'under the hood'. We'll also take a
look at how the metric :mod:`decorator API <.metrics.decorators>` can be used to construct powerful metrics which report running
and terminal statistics. Finally, we'll take a closer look at the :class:`.MetricTree` and :class:`.MetricList` which
make all of this happen internally.

Default Keys
------------------------------------

In typical usage of torchbearer, we rarely interface directly with the metric API, instead just providing keys to the
Model such as 'acc' and 'loss'. These keys are managed in a dict maintained by the decorator
:func:`default_for_key(key) <.default_for_key>`. Inside the torchbearer model, metrics are stored in an instance of
:class:`.MetricList`, which is a wrapper that calls each metric in turn, collecting the results in a dict. If
:class:`.MetricList` is given a string, it will look up the metric in the default metrics dict and use that instead. If
you have defined a class that implements :class:`.Metric` and simply want to refer to it with a key, decorate it with
:func:`.default_for_key`.

Metric Decorators
------------------------------------

Now that we have explained some of the basic aspects of the metric API, lets have a look at an example:

.. literalinclude:: /../torchbearer/metrics/primitives.py
   :lines: 22-26

This is the definition of the default accuracy metric in torchbearer, let's break it down.

:func:`.mean`, :func:`.std` and :func:`.running_mean` are all decorators which collect statistics about the underlying
metric. :class:`.CategoricalAccuracy` simply returns a boolean tensor with an entry for each item in a batch. The
:func:`.mean` and  :func:`.std` decorators will take a mean / standard deviation value over the whole epoch (by keeping
a sum and a number of values). The :func:`.running_mean` will collect a rolling mean for a given window size. That is,
the running mean is only computed over the last 50 batches by default (however, this can be changed to suit your needs).
Running metrics also have a step size, designed to reduce the need for constant computation when not a lot is changing.
The default value of 10 means that the running mean is only updated every 10 batches.

Finally, the :func:`.default_for_key` decorator is used to bind the metric to the keys 'acc' and 'accuracy'.

Lambda Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One decorator we haven't covered is the :func:`.lambda_metric`. This decorator allows you to decorate a function instead
of a class. Here's another possible definition of the accuracy metric which uses a function:

.. code-block:: python

   @metrics.default_for_key('acc')
   @metrics.running_mean
   @metrics.std
   @metrics.mean
   @metrics.lambda_metric('acc', on_epoch=False)
   def categorical_accuracy(y_pred, y_true):
      _, y_pred = torch.max(y_pred, 1)
      return (y_pred == y_true).float()

The :func:`.lambda_metric` here converts the function into a :class:`.MetricFactory`. This can then be used in the
normal way. By default and in our example, the lambda metric will execute the function with each batch of output
(y_pred, y_true). If we set `on_epoch=True`, the decorator will use an :class:`.EpochLambda` instead of a
:class:`.BatchLambda`. The :class:`.EpochLambda` collects the data over a whole epoch and then executes the metric at
the end.

Metric Output - to_dict
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At the root level, torchbearer expects metrics to output a dictionary which maps the metric name to the value. Clearly,
this is not done in our accuracy function above as the aggregators expect input as numbers / tensors instead of
dictionaries. We could change this and just have everything return a dictionary but then we would be unable to tell the
difference between metrics we wish to display / log and intermediate stages (like the tensor output in our example
above). Instead then, we have the :func:`.to_dict` decorator. This decorator is used to wrap the output of a metric in a
dictionary so that it will be picked up by the loggers. The aggregators all do this internally (with 'running\_',
'\_std', etc. added to the name) so there's no need there, however, in case you have a metric that outputs precisely the
correct value, the :func:`.to_dict` decorator can make things a little easier.

Data Flow - The Metric Tree
------------------------------------

Ok, so we've covered the :mod:`decorator API <.metrics.decorators>` and have seen how to implement all but the most
complex metrics in torchbearer. Each of the decorators described above can be easily associated with one of the metric
aggregator or wrapper classes so we won't go into that any further. Instead we'll just briefly explain the
:class:`.MetricTree`. The :class:`.MetricTree` is a very simple tree implementation which has a root and some children.
Each child could be another tree and so this supports trees of arbitrary depth. The main motivation of the metric tree
is to co-ordinate data flow from some root metric (like our accuracy above) to a series of leaves (mean, std, etc.).
When :meth:`.Metric.process` is called on a :class:`.MetricTree`, the output of the call from the root is given to each
of the children in turn. The results from the children are then collected in a dictionary. The main reason for including
this was to enable encapsulation of the different statistics without each one needing to compute the underlying metric
individually. In theory the :class:`.MetricTree` means that vastly complex metrics could be computed for specific use
cases, although I can't think of any right now...
