=================================
DCRL-Green's custom DC simulation
================================= 

Dependencies
------------

1. Linux OS (Ubuntu 20.04)
2. Conda

First time setup
----------------

Clone the latest DCRL-Green version from github using:

.. code-block:: bash
    
    git clone https://github.com/HewlettPackard/dc-rl.git

If using SSH, execute:

.. code-block:: bash
    
    git clone git@github.com:HewlettPackard/dc-rl.git

Change the current working directory to the dc-rl folder:

.. code-block:: bash
    
    cd dc-rl

Create a conda environment and install dependencies:

.. code-block:: bash
    
    conda create -n dcrl python=3.10
    conda activate dcrl
    pip install -r requirements.txt
