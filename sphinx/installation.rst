============
Installation
============

|F| enables data center simulation framework hosted using OpenAI Gym. The framework is self-contained but allows for the integration of other co-simulation models through a plug-in interface.
|F|'s simulation model provides users with advanced customization options for DC designs, including the ability to specify individual servers within racks. More details on the framework configuration is provided in the :ref:`Environment` section. Follow the steps below to setup your |F| environment.

Dependencies
------------

1. Linux OS (tested on Ubuntu 20.04)
2. Conda_

.. _Conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html

First time setup
----------------

Clone the latest |F| version from GitHub using:

.. code-block:: bash
    
    git clone https://github.com/HewlettPackard/sustaindc.git

If using SSH, execute:

.. code-block:: bash
    
    git clone git@github.com:HewlettPackard/sustaindc.git

Change the current working directory to the sustaindc folder:

.. code-block:: bash
    
    cd sustaindc

Create a conda environment and install the needed dependencies:

.. code-block:: bash
    
    conda create -n sustaindc python=3.10
    conda activate sustaindc
    pip install -r requirements.txt

