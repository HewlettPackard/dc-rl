============
Installation
============

Follow the steps below to setup your |F| environment.

Dependencies
------------

1. Linux OS (tested on Ubuntu 20.04)
2. Python 3.7+
3. Conda_ (optional)
4. Dependencies listed in the `requirements.txt` file

.. _Conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html

Installation Steps
--------------------

Clone the latest |F| version from GitHub using:

.. code-block:: bash
    
    git clone https://github.com/HewlettPackard/dc-rl.git

If using SSH, execute:

.. code-block:: bash
    
    git clone git@github.com:HewlettPackard/dc-rl.git

Navigate to the project repository:

.. code-block:: bash
    
    cd dc-rl

Create a Conda_ environment if you prefer using a virtual Python environment to manage packages for this project (optional):

.. code-block:: bash
    
    conda create -n sustaindc python=3.10
    conda activate sustaindc


Install the required packages:

.. code-block:: bash

    pip install -r requirements.txt


