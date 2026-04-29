from setuptools import setup, Extension
import pybind11
import os

cpp_args = ['/arch:AVX2', '/openmp'] if os.name == 'nt' else ['-mavx2', '-fopenmp']

s_ext = Extension(
    'engine_bridge',
    sources=['analog_universal_node_engine_avx2.cpp', 'bindings.cpp'],
    include_dirs=[pybind11.get_include(), '.'],
    library_dirs=['ftt'],
    libraries=['libfftw3-3'],
    language='c++',
    extra_compile_args=cpp_args,
)

setup(
    name='engine_bridge',
    version='0.1',
    ext_modules=[s_ext],
)
