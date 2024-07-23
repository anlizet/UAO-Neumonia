import unittest
import numpy as np
from detector_neumonia import App, predict

class TestPredictFunction(unittest.TestCase):
    def test_predict_return_types(self):
        label, proba, heatmap = predict(np.array([]))
        self.assertIsInstance(label, str)
        self.assertIsInstance(proba, float)
        self.assertIsInstance(heatmap, np.ndarray)
        self.assertEqual(heatmap.shape, (512, 512))

    def test_predict_probability_range(self):
        _, proba, _ = predict(np.array([]))
        self.assertGreaterEqual(proba, 80)
        self.assertLessEqual(proba, 99)

if __name__ == '__main__':
    unittest.main()
