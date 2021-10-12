.. _API reference:

=============
API reference
=============

.. contents::
   :local:

Decorators
==========

Instruction
-----------
.. autofunction:: asr.instruction

CLI constructors
----------------
.. autofunction:: asr.option

.. autofunction:: asr.argument

Migration
---------
.. autofunction:: asr.migration

Dataclasses
===========

Record
------
.. autoclass:: asr.Record
   :members:

Run Specification
-----------------
.. autoclass:: asr.RunSpecification
   :members:

Resources
---------
.. autoclass:: asr.Resources
   :members:

Dependencies
------------
.. autoclass:: asr.Dependencies
   :members:

History
-------
.. autoclass:: asr.RevisionHistory
   :members:

Metadata
--------
.. autoclass:: asr.Metadata
   :members:


Database sub-package
====================

Run Application
---------------
.. autofunction:: asr.database.run_app


Application object
------------------
.. autoclass:: asr.database.App
   :members:

Database project
----------------
.. autoclass:: asr.database.DatabaseProject
   :members: