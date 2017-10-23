

API Reference
=============

This is the class and function reference of batman. Please refer to
previous sections for further details, as the class and function raw
specifications may not be enough to give full guidelines on their uses.

:mod:`batman.space`: Parameter space
------------------------------------

.. .. automodule:: batman.space
..    :no-members:
..    :no-inherited-members:

.. currentmodule:: batman

.. autosummary::

   space.Point
   space.Space
   space.Doe
   space.Refiner

.. py:module:: space
.. automodule:: batman.space
   :members:
   :undoc-members:

:mod:`batman.surrogate`: Surrogate Modelling
--------------------------------------------

.. currentmodule:: batman

.. autosummary::

   surrogate.SurrogateModel
   surrogate.Kriging
   surrogate.PC
   surrogate.RBFnet

.. py:module:: surrogate
.. automodule:: batman.surrogate
   :members:
   :undoc-members:

:mod:`batman.uq`: Uncertainty Quantification
--------------------------------------------

.. currentmodule:: batman

.. autosummary::

   uq.UQ

.. py:module:: uq
.. automodule:: batman.uq
   :members:
   :undoc-members:

:mod:`batman.visualization`: Uncertainty Visualization
------------------------------------------------------

.. currentmodule:: batman

.. autosummary::

   visualization.Kiviat3D
   visualization.HdrBoxplot
   visualization.doe
   visualization.response_surface
   visualization.sobol
   visualization.corr_cov
   visualization.pdf
   visualization.kernel_smoothing
   visualization.reshow

.. py:module:: visualization
.. automodule:: batman.visualization
   :members:
   :undoc-members:

:mod:`batman.pod`: Proper Orthogonal Decomposition
--------------------------------------------------

.. currentmodule:: batman

.. autosummary::

   pod.Core
   pod.Pod

.. py:module:: pod
.. automodule:: batman.pod
   :members:
   :undoc-members:

:mod:`batman.functions`: Functions
----------------------------------

.. currentmodule:: batman

.. autosummary::

   functions.SixHumpCamel
   functions.Branin
   functions.Michalewicz
   functions.Ishigami
   functions.Rastrigin
   functions.G_Function
   functions.Forrester
   functions.Channel_Flow
   functions.Manning
   functions.Mascaret
   functions.MascaretApi
   functions.ChemicalSpill
   functions.multi_eval
   functions.output_to_sequence

.. py:module:: functions
.. automodule:: batman.functions.analytical
   :members:
   :undoc-members:

.. automodule:: batman.functions.mascaret
   :members:
   :undoc-members:

:mod:`batman.tasks`: Tasks
--------------------------

.. currentmodule:: batman

.. autosummary::

   tasks.SnapshotTask
   tasks.SnapshotProvider
   tasks.Snapshot

.. py:module:: tasks
.. automodule:: batman.tasks
   :members:
   :undoc-members:

:mod:`batman.misc`: Misc
------------------------

.. currentmodule:: batman

.. autosummary::

   misc.NestedPool
   misc.ProgressBar
   misc.optimization
   misc.import_config
   misc.check_yes_no
   misc.ask_path
   misc.abs_path
   misc.clean_path

.. py:module:: misc
.. automodule:: batman.misc
   :members:
   :undoc-members:

:mod:`batman.input_output`: Input Output
----------------------------------------

.. currentmodule:: batman

.. autosummary::

   input_output.Dataset
   input_output.IOFormatSelector

.. py:module:: input_output
.. automodule:: batman.input_output
   :members:
   :undoc-members:

