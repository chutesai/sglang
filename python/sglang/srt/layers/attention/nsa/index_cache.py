"""IndexCache: Cross-layer index reuse for the DSA lightning indexer.

Designates layers as "Full" (run indexer normally) or "Shared" (reuse indices
from the nearest preceding Full layer), eliminating up to 75% of indexer
computations with negligible quality loss.

Reference: arxiv 2603.12201
"""

from __future__ import annotations

import json
import logging
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)


class IndexCacheConfig:
    """Determines which layers are Full vs Shared for IndexCache."""

    full_layers: Set[int]
    shared_layers: Set[int]
    shared_to_full: Dict[int, int]

    def __init__(
        self,
        num_layers: int,
        config_path: Optional[str] = None,
        ratio: Optional[float] = None,
    ):
        if config_path:
            with open(config_path) as f:
                data = json.load(f)
            self.full_layers = set(data["full_layers"])
        elif ratio is not None and ratio < 1.0:
            # Uniform spacing: keep every 1/ratio layers as Full
            step = max(1, int(round(1.0 / ratio)))
            self.full_layers = set(range(0, num_layers, step))
        else:
            # All layers are Full = IndexCache disabled
            self.full_layers = set(range(num_layers))

        # Layer 0 must always be Full — shared layers reuse indices from
        # preceding Full layers, so there must be a Full layer before any
        # Shared layer. Without layer 0 as Full, the first Shared layers
        # would read uninitialized indices and crash.
        if 0 not in self.full_layers:
            logger.warning(
                "IndexCache: layer 0 was not marked as Full. "
                "Forcing layer 0 to Full (required for correctness)."
            )
            self.full_layers.add(0)

        self.shared_layers = set(range(num_layers)) - self.full_layers
        # Map each shared layer to its nearest preceding full layer
        self.shared_to_full = {}
        sorted_full = sorted(self.full_layers)
        for layer_id in sorted(self.shared_layers):
            preceding = [f for f in sorted_full if f < layer_id]
            assert preceding, (
                f"Bug: shared layer {layer_id} has no preceding Full layer "
                f"despite layer 0 being Full."
            )
            self.shared_to_full[layer_id] = preceding[-1]

        logger.info(
            f"IndexCache: {len(self.full_layers)} Full layers, "
            f"{len(self.shared_layers)} Shared layers "
            f"(ratio={len(self.full_layers)/num_layers:.2f}). "
            f"Full layers: {sorted(self.full_layers)}"
        )

    def is_full_layer(self, layer_id: int) -> bool:
        return layer_id in self.full_layers

    def is_shared_layer(self, layer_id: int) -> bool:
        return layer_id in self.shared_layers
