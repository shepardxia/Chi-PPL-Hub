"""Mock adapter returning the groundtruth_code as the generation.

Sanity check: with this adapter, scores should be ~perfect.
"""

from .base import GenerationResult


class GroundTruthAdapter:
    name = "groundtruth"

    def generate(self, atom: dict) -> GenerationResult:
        return GenerationResult(
            code=atom["groundtruth_code"],
            raw_response="(groundtruth)",
        )
