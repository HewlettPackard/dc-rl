============================================
EnergyPlus enabled DC simulation (DCRLeplus)
============================================

DCRL-Green uses Sinergym_ :cite:p:`2021sinergym` to create an environment following the OpenAI Gym :cite:p:`brockman2016openai` interface for wrapping EnergyPlus for data center control. 
It is recommended to install Sinergym supported by a Docker container that imports dependencies by default. The instructions for manual installations are also provided in the following sections.

*******************
1. Installation via Docker image (**recommended**)
*******************

1.1 Prerequisites
-----------------

Install Docker_

.. _Sinergym: https://ugr-sail.github.io/sinergym/compilation/main/index.html
.. _Docker: https://docs.docker.com/get-docker/


1.2 First-time setup
--------------------

Clone the latest DCRL-Green version from GitHub using:

.. code-block:: bash
    
    git clone https://github.com/HewlettPackard/dc-rl.git

Pull the Docker image with command:

.. code-block:: bash
    
    docker pull agnprz/carbon_sustain:v3

.. note::
   To run command with elevated privileges, use the prefix :code:`sudo` 


**********************
2. Manual installation (**not recommended**)
**********************

For manual installation (not recommended), you need to follow the instructions to manually install `Sinergym <https://ugr-sail.github.io/sinergym/compilation/main/pages/installation.html#manual-installation>`_. Make sure all the required components (custom Python environment, EnergyPlus, and BCVTB) are correctly installed.