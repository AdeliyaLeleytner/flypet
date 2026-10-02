import unittest

try:
    import torch
except ImportError:  # language-model extras are optional, see requirements-llm.txt
    raise unittest.SkipTest("torch is not installed")
from scripts.train_question_decoder import teacher_inputs, inference_inputs, summary


class TrainingTests(unittest.TestCase):
    def test_teacher_mask_and_inference_target_exclusion(self):
        prefix = torch.randn(2, 8, 5, requires_grad=True)
        questions = torch.randn(3, 7, 5)
        answers = torch.tensor([[5, 9], [6, 9], [7, 9], [8, 9]])
        tails = torch.randn(3, 4, 9, 5)
        q = torch.tensor([0, 2])
        y = torch.tensor([1, 3])
        x, label = teacher_inputs(prefix, q, y, tails, answers)
        self.assertEqual(x.shape, (2, 17, 5))
        self.assertTrue((label[:, :15] == -100).all())
        torch.testing.assert_close(label[:, -2:], answers[y])
        raw = inference_inputs(prefix, q, questions)
        self.assertEqual(raw.shape, (2, 15, 5))
        torch.testing.assert_close(raw[:, :8], prefix)
        torch.testing.assert_close(raw[:, 8:], questions[q])
        x.sum().backward()
        torch.testing.assert_close(prefix.grad, torch.ones_like(prefix))

    def test_summary_preserves_grouped_both_questions(self):
        rows = []
        for assignment in [0, 1]:
            for i in [0, 1]:
                rows.append(
                    {
                        "group": "g",
                        "assignment": assignment,
                        "correct": assignment == 0,
                        "valid": True,
                        "first_answer_token_nll": 1.0,
                        "query_is_last_mentioned": bool(i),
                    }
                )
        x = summary(rows)
        self.assertEqual(x["all"]["accuracy"], 0.5)
        self.assertEqual(x["both_queries_correct"], 0.5)


if __name__ == "__main__":
    unittest.main()
