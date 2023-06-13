=================================
DCRL-Green's custom DC simulation
================================= 

Dependencies
------------

1. Python_ 
    Python version 3.x is recommended.

.. _Python: https://python.land/installing-python

2. `Gymnasium <gym>`_ 
    Enivronments defined with DCRL-Green can easily be wrapped by Gymnasium interface.

.. _gym: https://github.com/Farama-Foundation/Gymnasium

3. `Ray RLlib <ray>`_
    DCRL-Green uses RLlib to execute reinforcement learning algorithms.

.. _ray: https://docs.ray.io/en/latest/rllib/index.html

4. TensorFlow and PyTorch
    RLlib does not automatically install a deep-learning framework, but supports TensorFlow (both 1.x with static-graph and 2.x with eager mode) as well as PyTorch. Depending on your needs, make sure to install either TensorFlow or PyTorch (or both).

5. TensorBoard_ 
    By default, RLlib logs results in Tensorboard. 
    
.. _TensorBoard: https://pypi.org/project/tensorboard/

First time setup
----------------

Clone the latest DCRL-Green version from github using:

.. code-block:: bash
    
    git clone https://github.com/HewlettPackard/dc-rl.git