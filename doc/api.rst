

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

   space.Sample
   space.Space
   space.gp_sampler.GpSampler
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
   surrogate.SklearnRegressor
   surrogate.Evofusion
   surrogate.Mixture

.. py:module:: surrogate
.. automodule:: batman.surrogate
   :members:
   :undoc-members:

:mod:`batman.uq`: Uncertainty Quantification
--------------------------------------------

.. currentmodule:: batman

.. autosummary::

   uq.UQ
   uq.cosi

.. py:module:: uq
.. automodule:: batman.uq
   :members:
   :undoc-members:

:mod:`batman.visualization`: Uncertainty Visualization
------------------------------------------------------

.. currentmodule:: batman

.. autosummary::

   visualization.Kiviat3D
   visualization.Tree
   visualization.HdrBoxplot
   visualization.doe
   visualization.response_surface
   visualization.sensitivity_indices
   visualization.cusunoro
   visualization.moment_independent
   visualization.corr_cov
   visualization.pdf
   visualization.kernel_smoothing
   visualization.reshow
   visualization.mesh_2D

.. py:module:: visualization
.. automodule:: batman.visualization
   :members:
   :undoc-members:

:mod:`batman.pod`: Proper Orthogonal Decomposition
--------------------------------------------------

.. currentmodule:: batman

.. autosummary::

   pod.Pod

.. py:module:: pod
.. automodule:: batman.pod
   :members:
   :undoc-members:

:mod:`batman.functions`: Functions
----------------------------------

.. currentmodule:: batman

.. autosummary::

   functions.data
   functions.DbGeneric
   functions.analytical.SixHumpCamel
   functions.analytical.Branin
   functions.analytical.Michalewicz
   functions.analytical.Ishigami
   functions.analytical.Rastrigin
   functions.analytical.G_Function
   functions.analytical.Forrester
   functions.analytical.ChemicalSpill
   functions.analytical.Channel_Flow
   functions.analytical.Manning
   functions.db_Mascaret
   functions.utils.multi_eval
   functions.utils.output_to_sequence

.. py:module:: functions
.. automodule:: batman.functions.data
   :members:
   :undoc-members:

.. automodule:: batman.functions.analytical
   :members:
   :undoc-members:

.. automodule:: batman.functions.db_Mascaret
   :members:
   :undoc-members:

:mod:`batman.tasks`: Tasks
--------------------------

.. currentmodule:: batman

.. autosummary::

   tasks.ProviderFunction
   tasks.ProviderFile
   tasks.ProviderJob

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

   input_output.available_formats
   input_output.formater

.. py:module:: input_output
.. automodule:: batman.input_output
   :members:
   :undoc-members:

