from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

setup(ext_modules=cythonize('kerrpy_cython/*.pyx'),
      include_dirs=[numpy.get_include()]
      )

extensions = [
    Extension("geodesic_integrator", ["kerrpy_cython/geodesic_integrator.pyx"],
              include_dirs=[numpy.get_include()]),
    Extension("metric", ["kerrpy_cython/metric.pyx"],
              include_dirs=[numpy.get_include()]),
    Extension("initial_setup", ["kerrpy_cython/initial_setup.pyx"],
              libraries=["m"],
              extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp"],
              extra_link_args=['-fopenmp'],
              include_dirs=[numpy.get_include()]),
]

setup(
    name="kerrpy_cython",
    ext_modules=cythonize(extensions),
)
