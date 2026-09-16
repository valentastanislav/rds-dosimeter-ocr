import contextlib
import io
import unittest

import dosimeter_get_values_flow as flow


class FlowHelpTests(unittest.TestCase):
    def test_help_is_available_without_required_runtime_arguments(self):
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            result = flow.main(["--help"])

        text = output.getvalue()

        self.assertEqual(result, 0)
        self.assertIn("--track-time", text)
        self.assertIn("--select-reference-box", text)
        self.assertIn("--select-quad", text)
        self.assertIn("--select-grid", text)
        self.assertIn("--decoder-strategy rds30-joint-spatial", text)
        self.assertIn("--rds200-pattern-refinement", text)
        self.assertIn("Geometry is video-specific", text)


if __name__ == "__main__":
    unittest.main()
