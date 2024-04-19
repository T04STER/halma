from setuptools import setup, Extension
import numpy as np

jump_moves = Extension(
    'jump_moves',
    sources = ['jump_moves.cpp'],
    include_dirs=[np.get_include()]
)
setup(
    name = 'JumpMovesGenerator',
    version='1.0',
    description = 'This is a package for generation of jump moves using DFS',
    ext_modules = [jump_moves]
    )
