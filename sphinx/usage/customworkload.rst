=================================
Custom Workload Data
=================================

By default, |F| includes workload traces from `Alibaba <https://github.com/alibaba/clusterdata>`_ and `Google <https://github.com/google/cluster-data>`_ data centers. These traces are used to simulate the tasks that the DC needs to process, providing a realistic and dynamic workload for benchmarking purposes.

Data Source
-------------------

The default workload traces are extracted from:

  - Alibaba 2017 CPU Data (`LINK <https://github.com/alibaba/clusterdata>`__)
  - Google 2011 CPU Data (`LINK <https://github.com/google/cluster-data>`__)

Expected File Format
-----------------------

Workload trace files should be in :code:`.csv` format, with two columns: a timestamp or index (must be unnamed), and the corresponding DC Utilization (:code:`cpu_load`). The CPU load must be expressed as a fraction of the DC utilization (between 0 and 1). The workload file must contain one year of data with an hourly periodicity (365*24=8760 rows). 

Example Workload Trace File
--------------------------------

.. code-block:: python

   ,cpu_load
   1,0.380
   2,0.434
   3,0.402
   4,0.485
   ...

   
   


Integration Steps
----------------------
  - Place the new workload trace file in the :code:`data/Workload` folder
  - Update the :code:`workload_file` entry in :code:`DEFAULT_CONFIG` dictionary in :code:`sustaindc_env.py` with the path to the new workload trace file