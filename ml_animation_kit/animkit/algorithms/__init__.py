"""Shared building blocks for the animated algorithm diagrams: the base view, tree drawing, the network.

An algorithm diagram is a panel beside a dashboard's main plot that shows how the model computes. It
is redrawn from the live model at every training step, and for every example at inference. ``common``
holds the ``AlgorithmView`` base class, the signed blue/red colour rule and fitted-tree drawing;
``network_view`` draws a small neural network with every weight as an edge. Projects subclass these
views for their own models.
"""
