import unittest
from flypet.response_semantics import stated_direction


class SemanticsTests(unittest.TestCase):
    def test_current_is_not_a_temporal_change(self):
        self.assertIsNone(
            stated_direction(
                "The neural balance shifted toward avoidance relative to the reference."
            )
        )
        self.assertEqual(
            stated_direction("Approach is stronger in the current neural response."), 1
        )
        self.assertEqual(
            stated_direction("No tendency is stronger in the current neural response."), 0
        )
        self.assertEqual(stated_direction("My current neural response leans toward avoidance."), -1)

    def test_change_is_not_current_dominance(self):
        self.assertIsNone(
            stated_direction("My current neural balance leans toward avoidance.", "comparison")
        )
        self.assertEqual(
            stated_direction(
                "The neural balance shifted toward avoidance relative to the reference.",
                "comparison",
            ),
            -1,
        )
        self.assertEqual(
            stated_direction("No change in balance is resolved at this precision.", "comparison"), 0
        )
        self.assertIsNone(stated_direction("Relative strength unclear."))


if __name__ == "__main__":
    unittest.main()
