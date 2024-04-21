from setuptools import setup, Extension
import numpy as np

moves = Extension(
    'moves',
    sources = ['moves.cpp'],
    include_dirs=[np.get_include()]
)
setup(
    name = 'MovesGenerator',
    version='1.0',
    description = 'This is a package for moves generation of neighbouring jump moves using DFS',
    ext_modules = [moves]
)
