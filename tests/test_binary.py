import unittest

from llm2jev import (
    BinaryQuestion,
    Choice,
    JevRequest,
    Noul,
    Score,
    compile_binary_questions,
)


class BinaryQuestionTests(unittest.TestCase):
    def test_validates_candidate_for_question_type(self) -> None:
        with self.assertRaises(ValueError):
            BinaryQuestion(
                question_id="severity",
                question_type="score",
                candidate="high",  # type: ignore[arg-type]
                context="message",
                objective=None,
                condition="High",
            )

    def test_validates_noul_candidate(self) -> None:
        with self.assertRaises(ValueError):
            BinaryQuestion(
                question_id="refund",
                question_type="noul",
                candidate="yes",
                context="message",
                objective=None,
                condition=None,
            )


class CompileBinaryQuestionsTests(unittest.TestCase):
    def test_compiles_noul_with_criteria_to_true_and_false_tasks(self) -> None:
        request = JevRequest(
            state={"message": "Please refund me."},
            model="jev-latest",
            questions={
                "refund": Noul(
                    instructions="Is a refund requested?",
                    criteria={
                        "true": "A refund is requested",
                        "false": "No refund is requested",
                    },
                ),
            },
        )

        tasks = compile_binary_questions(request)

        self.assertEqual(len(tasks), 2)
        self.assertEqual([task.question_id for task in tasks], ["refund", "refund"])
        self.assertTrue(all(task.question_type == "noul" for task in tasks))
        self.assertEqual([task.candidate for task in tasks], ["true", "false"])
        self.assertEqual(
            [task.condition for task in tasks],
            ["A refund is requested", "No refund is requested"],
        )

    def test_compiles_noul_without_criteria_to_one_true_task(self) -> None:
        request = JevRequest(
            state="Please refund me.",
            model="jev-latest",
            questions={"refund": Noul(instructions="Is a refund requested?")},
        )

        tasks = compile_binary_questions(request)

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].candidate, "true")
        self.assertIsNone(tasks[0].condition)

    def test_compiles_partial_noul_criteria_to_two_tasks(self) -> None:
        request = JevRequest(
            state="Please refund me.",
            model="jev-latest",
            questions={"refund": Noul(criteria={"true": "A refund is requested"})},
        )

        tasks = compile_binary_questions(request)

        self.assertEqual([task.candidate for task in tasks], ["true", "false"])
        self.assertEqual(
            [task.condition for task in tasks],
            ["A refund is requested", None],
        )

    def test_compiles_choice_once_per_option_in_order(self) -> None:
        request = JevRequest(
            state="My card was charged twice.",
            model="jev-latest",
            questions={
                "department": Choice(
                    instructions="Which team should handle this?",
                    criteria={
                        "billing": "Charges and payments",
                        "returns": None,
                    },
                ),
            },
        )

        tasks = compile_binary_questions(request)

        self.assertEqual([task.candidate for task in tasks], ["billing", "returns"])
        self.assertEqual(
            [task.condition for task in tasks],
            ["Charges and payments", None],
        )
        self.assertTrue(all(task.question_type == "choice" for task in tasks))

    def test_compiles_score_once_per_level_in_order(self) -> None:
        criteria = ["Low", {"severity": "Medium"}, ["High", "Blocking"]]
        request = JevRequest(
            state="Export failed.",
            model="jev-latest",
            questions={
                "severity": Score(
                    instructions={"task": "Rate severity"},
                    criteria=criteria,
                ),
            },
        )

        tasks = compile_binary_questions(request)

        self.assertEqual([task.candidate for task in tasks], [0, 1, 2])
        self.assertEqual(
            [task.condition for task in tasks],
            ["Low", {"severity": "Medium"}, ["High", "Blocking"]],
        )

    def test_preserves_question_and_candidate_order_for_mixed_request(self) -> None:
        request = JevRequest(
            state="A message",
            model="jev-latest",
            questions={
                "check": Noul(),
                "tone": Choice(criteria={"calm": None, "angry": None}),
                "severity": Score(criteria=["Low", "High"]),
            },
        )

        tasks = compile_binary_questions(request)

        self.assertEqual(
            [(task.question_id, task.candidate) for task in tasks],
            [
                ("check", "true"),
                ("tone", "calm"),
                ("tone", "angry"),
                ("severity", 0),
                ("severity", 1),
            ],
        )

    def test_compiled_tasks_do_not_share_mutable_request_content(self) -> None:
        state = {"messages": ["first"]}
        description = {"details": ["billing"]}
        request = JevRequest(
            state=state,
            model="jev-latest",
            questions={"department": Choice(criteria={"billing": description})},
        )

        task = compile_binary_questions(request)[0]
        state["messages"].append("second")
        description["details"].append("changed")

        self.assertEqual(task.context, {"messages": ["first"]})
        self.assertEqual(task.condition, {"details": ["billing"]})

    def test_rejects_choice_without_criteria(self) -> None:
        request = JevRequest(
            state="message",
            model="jev-latest",
            questions={"empty": Choice(criteria={})},
        )

        with self.assertRaises(ValueError):
            compile_binary_questions(request)

    def test_compiles_choice_with_all_candidates_on_each_task(self) -> None:
        request = JevRequest(
            state="Charged twice",
            model="jev-latest",
            questions={"department": Choice(criteria={"billing": "Payments", "technical": None})},
        )

        tasks = compile_binary_questions(request)

        self.assertEqual(
            [task.choices for task in tasks],
            [
                (("billing", "Payments"), ("technical", None)),
                (("billing", "Payments"), ("technical", None)),
            ],
        )


if __name__ == "__main__":
    unittest.main()
