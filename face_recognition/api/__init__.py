"""API module for face recognition system."""

from .app import create_app
from .models import *
from .endpoints import *

__all__ = ['create_app']