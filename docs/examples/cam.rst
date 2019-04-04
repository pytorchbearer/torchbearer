Visualising CNNs: The Class Appearance Model
============================================

In this example we will demonstrate the :class:`.ClassAppearanceModel` callback included in torchbearer. This implements
one of the most simple (and therefore not always the most successful) deep visualisation techniques, discussed in the
paper `Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps <https://arxiv.org/abs/1312.6034>`__.

Background
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The process to obtain Figure 1 from the paper is simple, given a particular target class :math:`c`, we use
back-propagation to obtain

:math:`\arg\!\max_I \; S_c(I) - \lambda\Vert I \Vert_2^2\;,`

where :math:`S_c(I)` is the un-normalised score of :math:`c` for the image :math:`I` given by the network. The
regularisation term :math:`\Vert I \Vert_2^2` is necessary to prevent the resultant image from becoming overly noisy.
More recent visualisation techniques use much more advanced regularisers to obtain smoother, more realistic images.

Loading the Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Since we are just running the callback on a pre-trained model, we don't need to load any data in this example. Instead,
we use torchvision to load an Inception V1 trained on ImageNet with the following:

.. literalinclude:: /_static/examples/cam.py
   :language: python
   :lines: 11-21

We need to include the `None` check as we will initialise the :class:`.Trial` without a dataloader, and so it will pass
`None` to the model forward.

Running with the Callback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using imaging callbacks, we commonly need to include an inverse transform to return the images to the right space.
For torchvision, ImageNet models we can use the following:

.. literalinclude:: /_static/examples/cam.py
   :language: python
   :lines: 5-8

Finally we can construct and run the :class:`.Trial` with:

.. literalinclude:: /_static/examples/cam.py
   :language: python
   :lines: 26-33

Here we create two :class:`.ClassAppearanceModel` instances which target the `lemon` and `cup` classes respectively.
Since the :class:`.ClassAppearanceModel` is an :class:`.ImagingCallback`, we use the imaging API to send each of these
to files. Finally, we evaluate the model for a single step to generate the results.

Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The results for the above code are given below. There some shapes which resemble a lemon or cup, however, not to the
same extent shown in the paper. Because of the simplistic regularisation and objective, this model is highly sensitive
to hyper-parameter choices. These results could almost certainly be improved with some more careful selection.

.. figure:: /_static/img/lemon.png
   :scale: 200 %
   :alt: Class Appearance Model of InceptionV1 for `lemon`

.. figure:: /_static/img/cup.png
   :scale: 200 %
   :alt: Class Appearance Model of InceptionV1 for `cup`

Source Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The source code for the example is given below:

 :download:`Download Python source code: cam.py </_static/examples/cam.py>`
