import io
import math
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import chaosys  # noqa: E402


class BindingTest(unittest.TestCase):
    def test_version(self):
        self.assertTrue(chaosys.version())

    def test_rk4_known(self):
        s = chaosys.rk4((1, 1, 1), 1000, 0.01)
        self.assertAlmostEqual(s.x, -4.902819483748812, places=9)
        self.assertAlmostEqual(s.y, -3.743407675271541, places=9)
        self.assertAlmostEqual(s.z, 24.691885987964351, places=9)

    def test_adaptive(self):
        s, accepted = chaosys.adaptive((1, 1, 1), 1.0, 1e-10, 1e-12)
        self.assertGreater(accepted, 0)
        self.assertTrue(all(math.isfinite(v) for v in (s.x, s.y, s.z)))

    def test_lyapunov(self):
        lam = chaosys.lyapunov((1, 1, 1), 0.005)
        self.assertGreater(lam, 0.7)
        self.assertLess(lam, 1.1)

    def test_lyapunov_report(self):
        report = chaosys.lyapunov_report((1, 1, 1), 0.005)
        self.assertEqual(report.segments, 200)
        self.assertEqual(len(report.segment_lambdas), 200)
        self.assertGreater(report.exponent, 0.7)
        self.assertLess(report.exponent, 1.1)
        self.assertGreaterEqual(report.stddev, 0.0)
        self.assertGreaterEqual(report.stderr, 0.0)
        self.assertIsInstance(report.converged, bool)
        self.assertAlmostEqual(report.exponent,
                               chaosys.lyapunov((1, 1, 1), 0.005),
                               places=12)

    def test_lyapunov_spectrum(self):
        spectrum = chaosys.lyapunov_spectrum((1, 1, 1), 0.005)
        self.assertEqual(len(spectrum), 3)
        self.assertGreater(spectrum[0], spectrum[1])
        self.assertGreater(spectrum[1], spectrum[2])
        self.assertGreater(spectrum[0], 0.7)
        self.assertLess(spectrum[0], 1.1)
        self.assertAlmostEqual(sum(spectrum), -(10.0 + 1.0 + 8.0 / 3.0),
                               places=2)

    def test_poincare(self):
        pts = chaosys.poincare((1, 1, 1), 0.005, transient=1000,
                              max_steps=10000, capacity=20, component=2,
                              level=20.0)
        self.assertTrue(pts)
        for p in pts:
            self.assertAlmostEqual(p.z, 20.0, places=9)

    def test_ensemble(self):
        final = chaosys.ensemble([(1, 1, 1), (1.1, 1, 1)], 10, 0.01, 2)
        self.assertEqual(len(final), 2)
        ref = chaosys.rk4((1, 1, 1), 10, 0.01)
        self.assertAlmostEqual(final[0].x, ref.x, places=9)

    def test_embedding(self):
        vec = chaosys.embed([0, 1, 2, 3, 4], 2, 1)
        self.assertEqual(len(vec), 4)
        self.assertEqual(vec[0], (0.0, 1.0))
        self.assertEqual(vec[-1], (3.0, 4.0))

    def test_correlation_dimension(self):
        s = chaosys.rk4((1, 1, 1), 1000, 0.01)
        series = []
        for _ in range(4000):
            s = chaosys.rk4(s, 1, 0.01)
            series.append(s.x)
        dim = chaosys.correlation_dimension(series, 4, 20, 1500)
        self.assertGreater(dim, 1.4)
        self.assertLess(dim, 2.4)

    def test_permutation_entropy(self):
        self.assertEqual(chaosys.permutation_entropy(list(range(100))), 0.0)
        s = chaosys.rk4((1, 1, 1), 1000, 0.01)
        series = []
        for _ in range(4000):
            s = chaosys.rk4(s, 1, 0.01)
            series.append(s.x)
        h = chaosys.permutation_entropy(series, 3, 1)
        self.assertGreater(h, 0.2)
        self.assertLessEqual(h, 1.0)

    def test_time_dependent_system(self):
        def duffing(t, x):
            return [x[1],
                    -0.2 * x[1] + x[0] - x[0] ** 3 + 0.3 * math.cos(t)]

        x = chaosys.rk4_time_system(2, duffing, (1, 1), 0.0, 2000, 0.01)
        self.assertTrue(all(math.isfinite(v) for v in x))
        lam = chaosys.lyapunov_time_system(2, duffing, (1, 1), 0.0, 0.01,
                                          1000, 50, 100)
        self.assertGreater(lam, 0.0)
        y, accepted = chaosys.adaptive_time_system(2, duffing, (1, 1), 0.0,
                                                  10.0)
        self.assertTrue(all(math.isfinite(v) for v in y))
        self.assertGreater(accepted, 0)

    def test_generic_analysis(self):
        def rossler(x):
            return [-x[1] - x[2], x[0] + 0.2 * x[1],
                    0.2 + x[2] * (x[0] - 5.7)]

        lam = chaosys.lyapunov_system(3, rossler, (1, 1, 1), 0.01,
                                     1000, 50, 100)
        self.assertGreater(lam, 0.0)
        report = chaosys.lyapunov_report_system(3, rossler, (1, 1, 1), 0.01,
                                               1000, 50, 100)
        self.assertEqual(report.segments, 100)
        self.assertAlmostEqual(report.exponent, lam, places=12)
        spectrum = chaosys.lyapunov_spectrum_system(3, rossler, (1, 1, 1),
                                                   0.01, 1000, 50, 100)
        self.assertEqual(len(spectrum), 3)
        self.assertGreater(spectrum[0], spectrum[1])
        pts = chaosys.poincare_system(3, rossler, (1, 1, 1), 0.01,
                                     1000, 20000, 8, 2, 5.0)
        self.assertTrue(pts)
        for p in pts:
            self.assertAlmostEqual(p[2], 5.0, places=9)
        final = chaosys.ensemble_system(3, rossler, [(1, 1, 1), (1.1, 1, 1)],
                                       10, 0.01, 1)
        self.assertEqual(len(final), 2)
        self.assertTrue(all(math.isfinite(v) for row in final for v in row))

    def test_generic_system(self):
        x, accepted = chaosys.adaptive_system(
            2, lambda q: [q[1], -q[0]], [1.0, 0.0], 10.0)
        self.assertGreater(accepted, 0)
        self.assertAlmostEqual(0.5 * (x[0] ** 2 + x[1] ** 2), 0.5, places=6)

    def test_negative_counts(self):
        with self.assertRaises(ValueError):
            chaosys.rk4((1, 1, 1), -1, 0.01)
        with self.assertRaises(ValueError):
            chaosys.lyapunov((1, 1, 1), 0.01, segments=-1)
        with self.assertRaises(ValueError):
            chaosys.poincare((1, 1, 1), 0.01, max_steps=-1)
        with self.assertRaises(ValueError):
            chaosys.ensemble([(1, 1, 1)], -1, 0.01)
        with self.assertRaises(ValueError):
            chaosys.rk4_system(1, lambda q: [-q[0]], [1.0], -1, 0.01)

    def test_callback_exception(self):
        def bad(x):
            raise RuntimeError("callback failure")

        with self.assertRaises(RuntimeError):
            chaosys.rk4_system(1, bad, [1.0], 10, 0.01)
        with self.assertRaises(RuntimeError):
            chaosys.adaptive_system(1, bad, [1.0], 1.0)
        with self.assertRaises(RuntimeError):
            chaosys.lyapunov_system(1, bad, [1.0], 0.01,
                                    transient=10, segment=10, segments=10)

        def bad_time(t, x):
            raise RuntimeError("time callback failure")

        with self.assertRaises(RuntimeError):
            chaosys.rk4_time_system(1, bad_time, [1.0], 0.0, 10, 0.01)

    def test_non_finite_series(self):
        with self.assertRaises(chaosys.ChaosysError):
            chaosys.embed([0.0, float("nan"), 2.0], 2, 1)
        with self.assertRaises(chaosys.ChaosysError):
            chaosys.permutation_entropy([0.0, 1.0, float("inf"), 3.0], 2, 1)

    def test_empty_ensemble(self):
        self.assertEqual(chaosys.ensemble([], 10, 0.01), [])
        self.assertEqual(chaosys.ensemble_system(2, lambda q: [q[1], -q[0]],
                                                 [], 10, 0.01), [])

    def test_to_csv(self):
        rows = [chaosys.State(1.0, 2.0, 3.0), (4.5, 5.5, 6.5)]
        stream = io.StringIO()
        written = chaosys.to_csv(stream, rows, header=("x", "y", "z"))
        self.assertEqual(written, 2)
        self.assertEqual(stream.getvalue(),
                         "x,y,z\n1.0,2.0,3.0\n4.5,5.5,6.5\n")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.csv")
            self.assertEqual(chaosys.to_csv(path, [(1, 2), (3, 4)]), 2)
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "1,2\n3,4\n")
        with self.assertRaises(ValueError):
            chaosys.to_csv(io.StringIO(), [(1, 2)], header=("a", "b", "c"))


if __name__ == "__main__":
    unittest.main()
