from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

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
    Extension("kerrpy_cython.kerr_equations", ["kerrpy_cython/kerr_equations.pyx"],
              extra_compile_args=["-O3", "-frecord-gcc-switches", "-ffast-math", "-march=native"],
              include_dirs=[numpy.get_include()]),
    Extension("kerrpy_cython.initial_setup", ["kerrpy_cython/initial_setup.pyx"],
              libraries=["m"],
              extra_compile_args=["-O3", "-frecord-gcc-switches", "-ffast-math", "-march=native", "-fopenmp"],
              extra_link_args=['-fopenmp'],
              include_dirs=[numpy.get_include()]),
]

common_includes = ["kerrpy_cython/common/*.pyx"]

compiler_directives ={'boundscheck': False,
        'embedsignature': True,
        'wraparound': False,
        "cdivision": True,
        "language_level":3}

setup(
    name="kerrpy_cython",
    ext_modules=cythonize(extensions + common_includes,annotate=True,
                          compiler_directives=compiler_directives),
)
