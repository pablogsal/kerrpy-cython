import pytest
import time
import numpy as np
import numpy.testing as npt
import itertools
import kerrpy_cython.initial_setup
from kerrpy_cython.initial_setup import generate_initial_conditions

from utils import compiled_with_openmp


@pytest.mark.parametrize("dimension,parallel", itertools.product([10, 50, 100, 150, 200], [True, False]))
def test_initial_conditions_different_dimensions(dimension, parallel):
    # GIVEN
    camera = {"r": 40, "theta": 1.511, "phi": 0, "focal_lenght": 20, "rows": dimension, "cols": dimension,
              "pixel_heigh": 16 / dimension, "pixel_width": 16 / dimension, "yaw": 0, "roll": 0, "pitch": 0}
    # WHEN
    initial_conditions, _ = np.array(generate_initial_conditions(camera, 0.5, parallel=parallel))

    # THEN
    initial_conditions = np.array(initial_conditions)
    r = initial_conditions[::5].reshape(dimension, dimension)
    theta = initial_conditions[1::5].reshape(dimension, dimension)
    phi = initial_conditions[2::5].reshape(dimension, dimension)
    pr = initial_conditions[3::5].reshape(dimension, dimension)
    ptheta = initial_conditions[4::5].reshape(dimension, dimension)

    # Test that all values of the coordinates are set to the coordinates of the camera
    npt.assert_equal(r, 40)
    npt.assert_equal(theta, 1.511)
    npt.assert_equal(phi, 0.0)

    half_image = dimension // 2
    # Test that pr is radially symmetric
    npt.assert_allclose(pr[:half_image, :][::-1], pr[half_image:, :])
    npt.assert_allclose(pr[:, :half_image][::-1], pr[:, :half_image])

    # Test that pr is antisymmetric along the y axis and is is increasing in absolute value
    npt.assert_allclose(ptheta[:half_image, :][::-1], -ptheta[half_image:, :])
    assert np.all(ptheta[:half_image, :] > 0)
    assert np.all(np.diff(ptheta[:half_image:, :][::-1], axis=0) > 0)
    assert np.all(ptheta[half_image:, :] < 0)
    assert np.all(np.diff(ptheta[half_image::, :][::-1], axis=0) > 0)


@pytest.mark.skipif(not compiled_with_openmp(kerrpy_cython.initial_setup),reason="Parallel version not available")
def test_initial_conditions_parallel_is_faster():
    # GIVEN
    dimension = 1000
    camera = {"r": 40, "theta": 1.511, "phi": 0, "focal_lenght": 20, "rows": dimension, "cols": dimension,
              "pixel_heigh": 16 / dimension, "pixel_width": 16 / dimension, "yaw": 0, "roll": 0, "pitch": 0}
    # WHEN
    t0 = time.perf_counter()
    np.array(generate_initial_conditions(camera, 0.5, parallel=False)[0])
    serial_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    initial_conditions = np.array(generate_initial_conditions(camera, 0.5, parallel=True)[0])
    parallel_time = time.perf_counter() - t0

    assert serial_time / parallel_time > 1.2
