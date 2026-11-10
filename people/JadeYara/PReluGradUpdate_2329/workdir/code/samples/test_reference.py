import unittest
import numpy as np
from reference import impl


class TrueTbeReferenceTest(unittest.TestCase):
    def test_finite_elementwise_not_reduced(self):
        g = np.array([[2, 3, 4, 5]], dtype=np.float32)
        x = np.array([[1, -2, 0, -0.0]], dtype=np.float32)
        w = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        dx, up = impl(g, x, w)
        np.testing.assert_array_equal(dx, [[2, 1.5, 2, 2.5]])
        np.testing.assert_array_equal(up, [[0, -6, 0, 0]])

    def test_positive_feature_does_not_hide_infinite_weight(self):
        one = np.ones((1, 1), np.float32)
        dx, up = impl(one, one, np.array([np.inf], np.float32))
        self.assertTrue(np.isnan(dx).all())
        self.assertTrue(np.isnan(up).all())

    def test_fp16_dx_overflow_propagates_into_update(self):
        dx, up = impl(np.array([[65504]], np.float16),
                      np.array([[-1]], np.float16), np.array([2], np.float16))
        self.assertTrue(np.isposinf(dx).all())
        self.assertTrue(np.isnan(up).all())

    def test_output_broadcast_and_spatial_weights(self):
        g = np.ones((2, 3, 4), np.float32)
        x = -np.ones((1, 3, 4), np.float32)
        w = np.arange(12, dtype=np.float32).reshape(3, 4)
        dx, up = impl(g, x, w)
        np.testing.assert_array_equal(dx, np.broadcast_to(w, (2, 3, 4)))
        np.testing.assert_array_equal(up, -np.ones((2, 3, 4)))

    def test_weights_cannot_expand_the_output(self):
        with self.assertRaises(ValueError):
            impl(np.ones((1, 1, 2), np.float32), np.ones((1, 1, 2), np.float32),
                 np.ones(3, np.float32))

    def test_nc1_uses_all_physical_lanes(self):
        g = np.ones((1, 2, 1, 1, 16), np.float32)
        dx, _ = impl(g, -g, np.arange(32, dtype=np.float32), "NC1HWC0")
        np.testing.assert_array_equal(dx.reshape(-1), np.arange(32))
        with self.assertRaises(ValueError):
            impl(g, -g, np.ones(19, np.float32), "NC1HWC0")

    def test_empty_output(self):
        x = np.empty((0, 3, 4), np.float32)
        self.assertEqual(impl(x, x, np.ones(3, np.float32))[0].shape, (0, 3, 4))


if __name__ == "__main__":
    unittest.main()
