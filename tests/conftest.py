"""
Shared pytest configuration.
Adds src/ to sys.path and sets environment variables before any test runs.
"""

import sys
import os

# Prevent TensorFlow/JAX crash on macOS (same guard as all src modules)
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"

# Make all src modules importable without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
