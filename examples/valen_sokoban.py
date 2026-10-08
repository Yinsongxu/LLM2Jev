"""Valen Sokoban evaluation using the current board as shared state."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any
from collections.abc import Mapping

from PIL import ImageDraw
from llm2jev import LLM2Jev, SGLangBackend, Choice, JevRequest
from llm2jev.inference.prompt import DefaultPromptRenderer


ACTIONS = ("up", "down", "left", "right")
DIRECTION_NAMES = {
    "en": {"up": "up", "down": "down", "left": "left", "right": "right"},
    "zh": {"up": "向上", "down": "向下", "left": "向左", "right": "向右"},
}


class SokobanPromptRenderer(DefaultPromptRenderer):
    """Render each Sokoban action without listing the other candidates."""

    @staticmethod
    def _choice_candidates(question) -> str:
        return ""


def _state_and_image(request: Mapping[str, Any], eval_dir: Path,
                     action_history: list[str], language: str):
    state = request.get("state")
    content: list[dict[str, Any]] = []
    for message in state["messages"]:
        for part in message.get("content", []):
            if part.get("type") == "text":
                content.append(part)
            elif part.get("type") == "image_url":
                image = part.get("image_url")
                image_path = (eval_dir / image["url"]).resolve()
                content.append({"type": "image_url", "image_url": {"url": image_path.as_uri()}})
    recent = action_history[-5:]
    if language == "en":
        history_text = "Recent moves: " + (", ".join(recent) if recent else "none")
    else:
        moves = "、".join(DIRECTION_NAMES[language][action] for action in recent)
        history_text = "最近动作：" + (moves if moves else "无（初始局面）")
    content.append({"type": "text", "text": history_text})
    return {"type": "multimodal", "content": content}


def record_to_request(request: Mapping[str, Any], eval_dir: Path, model: str,
                      language: str, action_history: list[str]) -> JevRequest:
    """Build a request with board evidence in state and actions in the question."""
    question = request["questions"]["next_action"]
    instructions = (
        "Which legal action best advances toward placing every box on a goal?"
        if language == "en"
        else "选择一个合法且最能推进箱子到达目标位置的动作。"
    )
    criteria = {
        action.capitalize(): DIRECTION_NAMES[language][action]
        for action in question["criteria"]
    }

    return JevRequest(
        model=model,
        state=_state_and_image(request, eval_dir, action_history, language),
        questions={
            "next_action": Choice(
                instructions=instructions,
                criteria=criteria,
            )
        },
    )


def make_gif(*, model_path: Path, eval_dir: Path, level_id: str,
             output: Path, max_steps: int) -> dict[str, object]:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from sokoban_eval.dataset import action_request
    from sokoban_eval.env import Board, SokobanEnv, State
    from sokoban_eval.render import render

    levels = [json.loads(line) for line in (eval_dir / "levels.jsonl").read_text(encoding="utf-8").splitlines()]
    level = next(item for item in levels if
     item["level_id"] == level_id)

    board = Board.from_dict(level["board"])
    initial = State.from_dict(level["initial_state"])
    env = SokobanEnv(board, initial, min(level["max_steps"], max_steps))
    output.parent.mkdir(parents=True, exist_ok=True)
    frames: list[ImageDraw.Image] = []
    decisions: list[str] = []

    with tempfile.TemporaryDirectory(prefix="llm2jev-valen-") as temporary:
        temporary_path = Path(temporary)
        device_options = {"mem_fraction_static": 0.75, "disable_cuda_graph": True,
                         "chunked_prefill_size": -1, "log_level": "error"}

        with SGLangBackend(model_path, submission="all", engine_kwargs=device_options) as backend:
            engine = LLM2Jev(backend=backend, renderer=SokobanPromptRenderer())

            while env.steps < env.max_steps and not board.solved(env.state):
                image_path = temporary_path / f"step-{env.steps:03d}.png"
                image = render(board, env.state, **level["render"])
                image.save(image_path)
                request = action_request(image_path.resolve(), level["language"])

                # LLM2Jev automatically evaluates all 4 actions
                response = engine.evaluate(record_to_request(
                    request, eval_dir, str(model_path), level["language"], decisions
                ))

                answer = response.answers["next_action"]

                # Select action with highest probability
                action = max(answer.probabilities.items(), key=lambda x: x[1])[0].lower()

                decisions.append(action)

                # Add info to frame
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, image.width, 48), fill="#111827")

                # Show objective and step
                boxes_remaining = len([b for b in env.state.boxes if b not in board.goals])
                draw.text((8, 5), f"Step {env.steps}   Boxes remaining: {boxes_remaining}", fill="white")

                # Show probabilities
                # Show probabilities for all actions
                prob_text = "  ".join(
                    f"{act}: {answer.probabilities.get(act.capitalize(), 0.0):.2f}" for act in ACTIONS
                )
                draw.text((8, 27), prob_text, fill="#d1d5db")
                frames.append(image)

                env.step(action)

            # Final frame
            final = render(board, env.state, **level["render"])
            draw = ImageDraw.Draw(final)
            draw.rectangle((0, 0, final.width, 40), fill="#111827")
            status = "🎉 SOLVED!" if board.solved(env.state) else "❌ Failed"
            boxes_on_goals = len([b for b in env.state.boxes if b in board.goals])
            total_boxes = len(env.state.boxes)
            draw.text((8, 6), f"{status}   Boxes on goals: {boxes_on_goals}/{total_boxes}", fill="white")
            draw.text((8, 22), f"Total steps: {env.steps}", fill="white")
            frames.append(final)

    frames[0].save(output, save_all=True, append_images=frames[1:], duration=650, loop=0, optimize=False)

    boxes_on_goals = len([b for b in env.state.boxes if b in board.goals])
    return {
        "level_id": level_id,
        "steps": env.steps,
        "solved": board.solved(env.state),
        "boxes_on_goals": f"{boxes_on_goals}/{len(env.state.boxes)}",
        "actions": decisions
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("/data/Qwen/Qwen3.5-9B"))
    parser.add_argument("--eval-dir", type=Path, default=Path("/data/benchmark/Valen-Eval-Game/eval_sokoban"))
    parser.add_argument("--level-id", default="sokoban-test-002")
    parser.add_argument("--output", type=Path, default=Path("assets/valen-sokoban.gif"))
    parser.add_argument("--max-steps", type=int, default=20)
    args = parser.parse_args()

    print("Using board state and concise action choices")
    print(f"Level: {args.level_id}, Max steps: {args.max_steps}\n")

    result = make_gif(
        model_path=args.model_path.resolve(),
        eval_dir=args.eval_dir.resolve(),
        level_id=args.level_id,
        output=args.output.resolve(),
        max_steps=args.max_steps,
    )

    print("\n" + "="*60)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("="*60)
    print(f"\n📁 GIF: {args.output.resolve()}")

    if result["solved"]:
        print(f"\n🎉🎉🎉 SUCCESS! Solved in {result['steps']} steps!")
    else:
        print(f"\n📊 Result: {result['boxes_on_goals']} boxes on goals")
        print(f"Actions: {' → '.join(result['actions'])}")

        from collections import Counter
        action_counts = Counter(result['actions'])
        print(f"\nAction distribution: {dict(action_counts)}")


if __name__ == "__main__":
    main()
