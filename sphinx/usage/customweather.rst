=================================
Custom Weather Data
=================================

Weather data captures the ambient environmental conditions that impact the DC's cooling requirements. |F| includes weather data files in the :code:`.epw` format from various locations where DCs are commonly situated.

Data Source
----------------

The default weather data files are extracted from:

- EnergyPlus Weather Data (`LINK <https://energyplus.net/weather>`_)

Expected File Format
---------------------------

Weather data files should be in the :code:`.epw` (EnergyPlus Weather) format, which includes hourly data for various weather parameters such as temperature, humidity, wind speed, etc.

Example Weather Data File
-------------------------------------

Below is a partial example of an :code:`.epw` file

.. code-block:: python 

   LOCATION,New York JFK Intl Ap NY USA,USA,NY,716070,40.63,-73.78,-5.0,3.4
   DESIGN CONDITIONS,1,New York JFK Intl Ap Ann Clg .4% Condns DB=>MWB,25.0
   ...
   DATA PERIODS,1,1,Data,Sunday, 1/ 1,12/31
   ...
   2022,1,1,1,60.1,45.1,1004.1,0,0,4.1,80.1,30.0
   2022,1,1,2,59.8,44.9,1004.0,0,0,4.1,80.0,29.8
   ...

Integration Steps
---------------------

  - Add the new weather data files in the .epw format to the :code:`data/Weather` folder
  - Update the :code:`weather_file` entry in :code:`DEFAULT_CONFIG` dictionary in :code:`sustaindc_env.py` with the path to the new weather file