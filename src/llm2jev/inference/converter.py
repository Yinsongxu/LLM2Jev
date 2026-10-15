from __future__ import annotations

from dataclasses import dataclass, field

from ..backend.base import Backend
from ..core.request import JevRequest
from ..core.response import JevResponse
from .assembler import Normalizer, assemble_response
from .normalization import normalize_l1
from .prompt import DefaultPromptRenderer, PromptRenderer
from .box import BOX_MAX_NEW_TOKENS


@dataclass(slots=True, kw_only=True)
class LLM2Jev:
    """Render, run inference, and assemble Jev requests."""

    backend: Backend
    renderer: PromptRenderer = field(default_factory=DefaultPromptRenderer)
    normalizer: Normalizer = normalize_l1

    def evaluate(self, request: JevRequest) -> JevResponse:
        rendered = self.renderer.render(request)
        generated = None
        if rendered.box_prompts:
            generated = self.backend.generate(
                model=request.model, prompts=rendered.box_prompts, max_new_tokens=BOX_MAX_NEW_TOKENS,
            )
        scored = None
        if rendered.score_prompts:
            scored = self.backend.score(model=request.model, prompts=rendered.score_prompts)
        return assemble_response(
            request=request,
            rendered=rendered,
            scored=scored,
            generated=generated,
            normalizer=self.normalizer,
        )
