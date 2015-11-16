useme
=======

A simple python package to design, run and calibrate models used in environmental sciences

What is useme?
~~~~~~~~~~~~~~~~

-  useme is a set of tools to create simple models and calibrate them via automatic optimizer
-

Installation
~~~~~~~~~~~~

``pip install useme`` or download the `source
code <https://bitbucket.org/jlerat/useme>`__ and
``python setup.py install``

Basic use
~~~~~~~~~


To setup a model, change its parameters and run it:

   .. code:: Python
        
       # Get an instance of the GR2M monthly rainfall-runoff model
       from useme.models import GR2M
       gr = GR2M()


More examples in the `examples folder <https://bitbucket.org/jlerat/useme/downloads>`__ directory.
