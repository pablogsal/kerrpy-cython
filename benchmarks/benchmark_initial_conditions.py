import pytest
import time
import numpy as np
import numpy.testing as npt
import itertools
import kerrpy_cython.initial_setup
from kerrpy_cython.initial_setup import get_initial_conditions

from tests.utils import compiled_with_openmp


@pytest.mark.parametrize("parallel", [True, False])
def test_initial_conditions_different_dimensions(parallel, benchmark):
    # GIVEN
    dimension = 1000
    camera = {"r": 40, "theta": 1.511, "phi": 0, "focal_lenght": 20, "rows": dimension, "cols": dimension,
              "pixel_heigh": 16 / dimension, "pixel_width": 16 / dimension, "yaw": 0, "roll": 0, "pitch": 0}
    # WHEN
    initial_conditions = benchmark(lambda :get_initial_conditions(camera, 0.5, parallel=parallel))


