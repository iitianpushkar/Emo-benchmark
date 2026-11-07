"""Check the binary class-balanced accuracy used for MOSEI emotion reporting."""

import unittest

try:
    import numpy as np
    from scripts.classification.mosei_mlp import metrics
except ImportError:
    np = None
    metrics = None


@unittest.skipIf(metrics is None, "ML dependencies are unavailable in this Python environment")
class WeightedAccuracyTests(unittest.TestCase):
    def test_per_emotion_weighted_accuracy_and_mean(self):
        # Each emotion has two positives and two negatives. One positive is
        # missed, while both negatives are correct: (1/2 + 2/2) / 2 = 0.75.
        y = np.tile(np.array([[1.0], [1.0], [0.0], [0.0]]), (1, 6))
        prob = np.tile(np.array([[0.9], [0.1], [0.2], [0.3]]), (1, 6))
        report = metrics(y, prob, np.full(6, 0.5))
        self.assertEqual(report["mean_weighted_accuracy"], 0.75)
        self.assertEqual([value["weighted_accuracy"] for value in report["per_label"].values()],
                         [0.75] * 6)


if __name__ == "__main__":
    unittest.main()
