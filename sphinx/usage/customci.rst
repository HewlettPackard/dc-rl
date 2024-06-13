=================================
Custom Carbon Intensity Data
=================================

Carbon Intensity (CI) data represents the carbon emissions associated with electricity consumption. |F| includes CI data files for various locations to simulate the carbon footprint of the DC's energy usage.

Data Source
-------------

The default carbon intensity data files are extracted from:

  - U.S. Energy Information Administration API (`LINK <https://api.eia.gov/bulk/EBA.zip>`_)

Expected File Format
------------------------

Carbon intensity files should be in :code:`.csv` format, with columns representing different energy sources and the average carbon intensity. The columns typically include:

  - :code:`timestamp`: The time of the data entry
  - :code:`WND`: Wind energy
  - :code:`SUN`: Solar energy
  - :code:`WAT`: Water (hydropower) energy
  - :code:`OIL`: Oil energy
  - :code:`NG`: Natural gas energy
  - :code:`COL`: Coal energy
  - :code:`NUC`: Nuclear energy
  - :code:`OTH`: Other energy sources
  - :code:`avg_CI`: The average carbon intensity

Example Carbon Intensity File
-----------------------------------------

.. code-block:: python

   timestamp,WND,SUN,WAT,OIL,NG,COL,NUC,OTH,avg_CI
   2022-01-01 00:00:00+00:00,1251,0,3209,0,15117,2365,4992,337,367.450
   2022-01-01 01:00:00+00:00,1270,0,3022,0,15035,2013,4993,311,363.434
   2022-01-01 02:00:00+00:00,1315,0,2636,0,14304,2129,4990,312,367.225
   2022-01-01 03:00:00+00:00,1349,0,2325,0,13840,2334,4986,320,373.228
   ...



Integration Steps
-----------------------

  - Add the new carbon intensity data files to the :code:`data/CarbonIntensity` folder
  - Update the :code:`cintensity_file` entry in :code:`DEFAULT_CONFIG` dictionary in :code:`sustaindc_env.py` with the path to the new carbon intensity file
