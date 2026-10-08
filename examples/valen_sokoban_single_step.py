"""Evaluate single-step visual Sokoban action choices against optimal labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from valen_sokoban import DIRECTION_NAMES, SokobanPromptRenderer, record_to_request
from sokoban_eval.env import ACTIONS, Board, State, transition
from llm2jev.backend.sglang import SGLangBackend
from llm2jev.core.questions import Choice
from llm2jev.core.request import JevRequest
from llm2jev.inference.binary import compile_binary_questions
from llm2jev.inference.normalization import normalize_l1
from llm2jev.inference.prompt import DefaultPromptRenderer


def evaluate(*, model_path: Path, eval_dir: Path, chunk_size: int,
             limit: int | None, task_type: str,
             order_rotations: int, renderer_type: str) -> dict[str, object]:
    records = [
        json.loads(line)
        for line in (eval_dir / "single_step.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if limit is not None:
        records = records[:limit]

    renderer = (
        SokobanPromptRenderer()
        if renderer_type == "no-candidates"
        else DefaultPromptRenderer()
    )
    prepared = []
    prompts = []
    candidate_order_count = 5 if task_type == "immediate-push" else 4
    if not 1 <= order_rotations <= candidate_order_count:
        raise ValueError(f"order_rotations must be in 1..{candidate_order_count}")

    for record_index, record in enumerate(records):
        language = record["meta"]["language_bucket"]
        request = record_to_request(
            record["request"], eval_dir, str(model_path), language, []
        )
        if task_type == "immediate-push":
            criteria = {
                action.capitalize(): DIRECTION_NAMES[language][action]
                for action in ("up", "down", "left", "right")
            }
            criteria["None"] = "none" if language == "en" else "没有"
            instructions = (
                "Which action will immediately push a box one cell? Choose None if no action does."
                if language == "en"
                else "哪个动作会立即推动一个箱子一格？如果没有这样的动作，选择“没有”。"
            )
            request = JevRequest(
                model=request.model,
                state=request.state,
                questions={"next_action": Choice(instructions=instructions, criteria=criteria)},
            )
        question = request.questions["next_action"]
        criteria_items = list(question.criteria.items())
        for rotation in range(order_rotations):
            offset = rotation % len(criteria_items)
            rotated_items = criteria_items[offset:] + criteria_items[:offset]
            rotated_request = JevRequest(
                model=request.model,
                state=request.state,
                questions={
                    "next_action": Choice(
                        instructions=question.instructions,
                        criteria=dict(rotated_items),
                    )
                },
            )
            tasks = compile_binary_questions(rotated_request)
            prepared.append((record_index, record, tasks, rotation))
            prompts.extend(renderer.render(task) for task in tasks)

    probabilities: list[float] = []
    device_options = {
        "mem_fraction_static": 0.75,
        "disable_cuda_graph": True,
        "chunked_prefill_size": -1,
        "log_level": "error",
    }
    with SGLangBackend(
        model_path, submission="all", engine_kwargs=device_options
    ) as backend:
        for start in range(0, len(prompts), chunk_size):
            batch = prompts[start:start + chunk_size]
            probabilities.extend(
                backend.score(model=str(model_path), prompts=batch).yes_probabilities
            )
            print(f"Scored {min(start + len(batch), len(prompts))}/{len(prompts)} prompts", flush=True)

    correct = legal = push_relevant = push_correct = predicted_pushes = 0
    per_language: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    action_counts: Counter[str] = Counter()
    top_probability_sum = margin_sum = 0.0
    optimal_count = 0
    position_counts: dict[int, Counter[int]] = defaultdict(Counter)
    rotation_stats: dict[int, list[int]] = defaultdict(lambda: [0, 0, 0])
    scores_by_record: list[dict[str, list[float]]] = [defaultdict(list) for _ in records]
    labels_by_record: list[list[str] | None] = [None for _ in records]
    cursor = 0

    for record_index, record, tasks, rotation in prepared:
        candidate_count = len(tasks)
        raw = probabilities[cursor:cursor + candidate_count]
        cursor += candidate_count
        normalized = normalize_l1(raw)
        labels = [str(task.candidate).lower() for task in tasks]
        if rotation == 0:
            labels_by_record[record_index] = labels
        for label, score in zip(labels, raw):
            scores_by_record[record_index][label].append(score)
        predicted = labels[max(range(candidate_count), key=lambda item: normalized[item])]
        position_counts[rotation][labels.index(predicted) + 1] += 1
        board = Board.from_dict(record["meta"]["board"])
        state = State.from_dict(record["meta"]["state"])
        if task_type == "immediate-push":
            target = {
                action for action in ACTIONS
                if transition(board, state, action)[2]
            } or {"none"}
        else:
            target = set(record["meta"]["optimal_actions"])
        is_optimal = predicted in target
        correct += int(is_optimal)
        optimal_count += len(target)
        rotation_stats[rotation][0] += int(is_optimal)
        rotation_stats[rotation][1] += len(target)
        rotation_stats[rotation][2] += 1

        language = record["meta"]["language_bucket"]
        per_language[language][0] += int(is_optimal)
        per_language[language][1] += 1
        action_counts[predicted] += 1

        if task_type == "optimal-action":
            _, moved, pushed = transition(board, state, predicted)
            legal += int(moved)
            predicted_pushes += int(pushed)
            optimal_push = any(transition(board, state, action)[2] for action in target)
            if optimal_push:
                push_relevant += 1
                push_correct += int(is_optimal and pushed)

        ordered = sorted(normalized, reverse=True)
        top_probability_sum += ordered[0]
        margin_sum += ordered[0] - ordered[1]

    evaluations = len(prepared)
    ensemble_correct = ensemble_target_count = 0
    ensemble_action_counts: Counter[str] = Counter()
    ensemble_languages: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    ensemble_top_probability = ensemble_margin = 0.0
    for record_index, record in enumerate(records):
        labels = labels_by_record[record_index]
        if labels is None:
            raise RuntimeError("missing base candidate order")
        averaged = {
            label: sum(scores_by_record[record_index][label]) / order_rotations
            for label in labels
        }
        ensemble_probabilities = normalize_l1([averaged[label] for label in labels])
        predicted = labels[max(range(len(labels)), key=lambda item: ensemble_probabilities[item])]
        board = Board.from_dict(record["meta"]["board"])
        state = State.from_dict(record["meta"]["state"])
        if task_type == "immediate-push":
            target = {
                action for action in ACTIONS
                if transition(board, state, action)[2]
            } or {"none"}
        else:
            target = set(record["meta"]["optimal_actions"])
        is_correct = predicted in target
        ensemble_correct += int(is_correct)
        ensemble_target_count += len(target)
        ensemble_action_counts[predicted] += 1
        language = record["meta"]["language_bucket"]
        ensemble_languages[language][0] += int(is_correct)
        ensemble_languages[language][1] += 1
        ordered = sorted(ensemble_probabilities, reverse=True)
        ensemble_top_probability += ordered[0]
        ensemble_margin += ordered[0] - ordered[1]

    return {
        "model": str(model_path),
        "task": task_type,
        "renderer": renderer_type,
        "records": len(records),
        "order_rotations": order_rotations,
        "evaluations": evaluations,
        "action_accuracy": correct / evaluations,
        "random_choice_baseline": optimal_count / (candidate_count * evaluations),
        "legal_action_rate": legal / evaluations if task_type == "optimal-action" else None,
        "optimal_push_cases": push_relevant if task_type == "optimal-action" else None,
        "optimal_push_action_accuracy_on_push_cases": (
            push_correct / push_relevant
            if task_type == "optimal-action" and push_relevant else None
        ),
        "predicted_push_rate": predicted_pushes / evaluations if task_type == "optimal-action" else None,
        "mean_top_probability": top_probability_sum / evaluations,
        "mean_top_two_margin": margin_sum / evaluations,
        "action_counts": dict(action_counts),
        "predicted_position_counts": {
            rotation: dict(counts) for rotation, counts in position_counts.items()
        },
        "per_rotation_accuracy": {
            rotation: {
                "accuracy": values[0] / values[2],
                "random_choice_baseline": values[1] / (candidate_count * values[2]),
            }
            for rotation, values in rotation_stats.items()
        },
        "permutation_ensemble": {
            "accuracy": ensemble_correct / len(records),
            "random_choice_baseline": (
                ensemble_target_count / (candidate_count * len(records))
            ),
            "action_counts": dict(ensemble_action_counts),
            "mean_top_probability": ensemble_top_probability / len(records),
            "mean_top_two_margin": ensemble_margin / len(records),
            "language_accuracy": {
                language: {
                    "correct": values[0],
                    "total": values[1],
                    "accuracy": values[0] / values[1],
                }
                for language, values in ensemble_languages.items()
            },
        },
        "language_accuracy": {
            language: {
                "correct": values[0],
                "total": values[1],
                "accuracy": values[0] / values[1],
            }
            for language, values in per_language.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("/data/Qwen/Qwen3.5-9B"))
    parser.add_argument(
        "--eval-dir", type=Path,
        default=Path("/data/benchmark/Valen-Eval-Game/eval_sokoban"),
    )
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--order-rotations", type=int, default=1)
    parser.add_argument(
        "--renderer", choices=("default", "no-candidates"), default="default"
    )
    parser.add_argument(
        "--task", choices=("optimal-action", "immediate-push"),
        default="optimal-action",
    )
    args = parser.parse_args()
    if (args.chunk_size < 1 or args.order_rotations < 1
            or (args.limit is not None and args.limit < 1)):
        parser.error("chunk-size, order-rotations, and limit must be positive")

    result = evaluate(
        model_path=args.model_path.resolve(),
        eval_dir=args.eval_dir.resolve(),
        chunk_size=args.chunk_size,
        limit=args.limit,
        task_type=args.task,
        order_rotations=args.order_rotations,
        renderer_type=args.renderer,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
