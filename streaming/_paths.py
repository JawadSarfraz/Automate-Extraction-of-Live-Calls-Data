"""Make the flat `ml/` scripts importable from the streaming package.

The ml/ directory is a set of flat scripts (not a package), and the existing code
already relies on inserting it on sys.path. We do the same here, in one place.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML = os.path.join(ROOT, "ml")
DATA = os.path.join(ROOT, "data")

if ML not in sys.path:
    sys.path.insert(0, ML)
