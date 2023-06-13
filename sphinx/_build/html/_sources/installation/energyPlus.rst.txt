================================
EnergyPlus enabled DC simulation
================================

DCRL-Green uses Sinergym_ :cite:p:`2021sinergym` to create an environment following Gymnasium interface for wrapping EnergyPlus for data center control. 

Prerequisites
-------------

Docker_
    We include a Dockerfile for installing and setting up the image for running Sinergym.

.. _Sinergym: https://ugr-sail.github.io/sinergym/compilation/main/index.html
.. _Docker: https://docs.docker.com/get-docker/


First-time setup
----------------

Clone the latest DCRL-Green version from github using:

.. code-block:: bash
    
    git clone https://github.com/HewlettPackard/dc-rl.git

Pull the Docker image with command:

.. code-block:: bash
    
    sudo docker pull agnprz/carbon_sustain:v3

A docker container can be launched using the command:

.. code-block:: console

   sudo docker run -t -i -v ./:/sinergym/dc-rl --shm-size=10.24gb agnprz/carbon_sustain:v3

If the docker container is launched successfully, the :code:`sinergym` environment is enabled. A python script can be executed by navigating to the directory :code:`dc-rl`.
