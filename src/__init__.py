"""Memory Over Maps.

Two-stage retrieval system for embodied object search: SigLIP2 feature
matching narrows candidates in milliseconds, then a VLM pass re-ranks
by fine-grained visibility. SAM3 segmentation + depth backprojection
produce a 3D object location for navigation.
"""

__version__ = "1.0.0"
