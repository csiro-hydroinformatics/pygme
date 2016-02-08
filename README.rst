pygme
=======

A simple python package to design, run and calibrate models used in environmental sciences

What is pygme?
~~~~~~~~~~~~~~~~

-  pygme is a set of tools to create simple models and calibrate them via automatic optimizer
-

Installation
~~~~~~~~~~~~

``pip install pygme`` or download the `source
code <https://bitbucket.org/jlerat/pygme>`__ and
``python setup.py install``

Basic use
~~~~~~~~~


To setup a model, change its parameters and run it:

   .. code:: Python
        
       # Get an instance of the GR2M monthly rainfall-runoff model
       from pygme.models import GR2M
       gr = GR2M()


More examples in the `examples folder <https://bitbucket.org/jlerat/pygme/downloads>`__ directory.
