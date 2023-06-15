============================================
EnergyPlus enabled DC simulation (DCRLeplus)
============================================

DCRL-Green uses Sinergym_ :cite:p:`2021sinergym` to create an environment following Gymnasium interface for wrapping EnergyPlus for data center control. 
It is recommended to install Sinergym supported by a Docker container that imports dependencies by default. The instructions for manual installations are also provided in the following sections.

*******************
1. Docker container
*******************

1.1 Prerequisites
-----------------

Docker_

.. _Sinergym: https://ugr-sail.github.io/sinergym/compilation/main/index.html
.. _Docker: https://docs.docker.com/get-docker/


1.2 First-time setup
--------------------

Clone the latest DCRL-Green version from github using:

.. code-block:: bash
    
    git clone https://github.com/HewlettPackard/dc-rl.git

Pull the Docker image with command:

.. code-block:: bash
    
    docker pull agnprz/carbon_sustain:v3

.. note::
   To run command with elevated privileges, use the prefix :code:`sudo` 

A docker container can be launched using the command:

.. code-block:: console

   docker run -t -i -v ./:/sinergym/dc-rl --shm-size=10.24gb agnprz/carbon_sustain:v3

If the docker container is launched successfully, the isolated :code:`sinergym` environment is enabled. A python script can be executed by navigating to the directory :code:`dc-rl`.

.. code-block:: console

   cd dc-rl

.. note::
   Some useful Docker commands could be found here_
   
.. _here: https://docs.docker.com/engine/reference/commandline/cli/

**********************
2. Manual installation
**********************