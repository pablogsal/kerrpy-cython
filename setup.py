from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("kerrpy_cython.geodesic_integrator", ["kerrpy_cython/geodesic_integrator.pyx"],
              extra_compile_args=["-O3", "-frecord-gcc-switches", "-ffast-math", "-march=native"],
              include_dirs=[numpy.get_include()]),
    Extension("kerrpy_cython.raytracer", ["kerrpy_cython/raytracer.pyx"],
              libraries=["m"],
              extra_compile_args=["-O3", "-frecord-gcc-switches", "-ffast-math", "-march=native", "-fopenmp"],
              extra_link_args=['-fopenmp'],
              include_dirs=[numpy.get_include()]),
    Extension("kerrpy_cython.metric", ["kerrpy_cython/metric.pyx"],
              extra_compile_args=["-O3", "-frecord-gcc-switches", "-ffast-math", "-march=native"],
              include_dirs=[numpy.get_include()]),
    Extension("kerrpy_cython.initial_setup", ["kerrpy_cython/initial_setup.pyx"],
              libraries=["m"],
              extra_compile_args=["-O3", "-frecord-gcc-switches", "-ffast-math", "-march=native", "-fopenmp"],
              extra_link_args=['-fopenmp'],
              include_dirs=[numpy.get_include()]),
]

common_includes = ["kerrpy_cython/_common/*.pyx"]

setup(
    name="kerrpy_cython",
    ext_modules=cythonize(extensions + common_includes),
)
