=================================
Usefull tips for the installation
=================================

Here you can see some usefull tips and what you should do if you get some specific errors.

Error 1: Failed to build wheels for cchardet
--------------------------------------------

You are probably missing a version of cython. Execute:

.. code-block:: bash

    pip install cython

Error 2: Microsoft Visual C++ 14.0 or greater is required
---------------------------------------------------------

You need to install Microsoft Visual C++ on your device.
You can do this on this website (https://visualstudio.microsoft.com/visual-cpp-build-tools/).
