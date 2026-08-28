import unittest

from Obselete.tmt_simulation import run_simulation


class TmtSimulationTests(unittest.TestCase):
    def test_run_simulation_returns_summary_and_history(self):
        result = run_simulation(
            mode="A",
            targets=["1", "2"],
            latency_params={"Goal Retrieval": 50, "Visual Search": 60, "Target Identification": 40, "Visuomotor Planning": 30, "Motor Execution": 50, "Performance Monitoring": 25, "Update Working Memory": 20},
            randomness=0.0,
            verbose=False,
        )

        self.assertEqual(result["mode"], "A")
        self.assertGreaterEqual(len(result["history"]), 2)
        self.assertGreater(result["summary"]["total_measured_ms"], 0)


if __name__ == "__main__":
    unittest.main()
