# micrograd/__init__.py
# MicroGrad — A minimal scalar-valued autograd engine built from scratch.
# Inspired by Andrej Karpathy's micrograd.

from micrograd.engine import Value, Neuron, Layer, MLP

__version__ = "0.1.0"
__author__ = "VanshAt"

__all__ = ["Value", "Neuron", "Layer", "MLP"]
