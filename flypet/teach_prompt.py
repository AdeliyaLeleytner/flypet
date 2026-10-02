"""The one prompt every teacher gets (frontier LLM, small LLM before and after GRPO)."""

from __future__ import annotations

from .teach import Task

SYSTEM = "You design training protocols for a fruit fly in a classical conditioning experiment. Answer with the protocol only."


def user_prompt(task: Task) -> str:
    return (
        "A fruit fly will be trained with classical conditioning and then tested on each odour.\n"
        f"Odours: {', '.join(task.panel)}.\n\n"
        "Goal for the test after training, compared with the same fly before training:\n"
        f"{task.describe()}\n\n"
        f"You may give at most {task.budget} trials. Each trial is one line in exactly one of these forms:\n"
        "TRAIN <odour> + REWARD    (the odour is paired with sugar)\n"
        "TRAIN <odour> + PUNISH    (the odour is paired with an electric shock)\n"
        "EXPOSE <odour>            (the odour alone, nothing else)\n\n"
        "Write only the trial lines, nothing else."
    )


def messages(task: Task) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_prompt(task)}]
