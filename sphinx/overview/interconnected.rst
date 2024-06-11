########################
Inter-connected Environments
########################

These three environments are interconnected to simulate realistic DC operations:

- The **Workload Environment** generates the computational demand that the **Data Center Environment** must process. This includes managing the scheduling of delayable tasks to optimize energy consumption and reduce the carbon footprint.

- The **Data Center Environment** handles the cooling and IT operations required to process the workloads. Higher computational demand results in increased heat generation, necessitating more cooling and energy consumption.

- The **Battery Environment** supports the DC by providing auxiliary energy during periods of high carbon intensity, helping to reduce the overall carbon footprint. It is affected by both the **Workload Environment** and the **Data Center Environment**. The workload impacts heat generation, which in turn affects the cooling requirements and energy consumption, influencing the battery's charging and discharging cycles.


Together, these interconnected environments provide a dynamic platform for benchmarking MARL algorithms, helping to develop strategies for more sustainable and efficient DC operations. A schematic of their connection is given in the figure below. 

.. image:: ../images/RLLoopv3.png
   :scale: 60 %
   :alt: RL loop
   :align: center
