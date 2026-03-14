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
            # Uniform spacing matching THUDM reference formula:
            # skip_topk = (max(layer_id-1, 0) % freq != 0)
            # This keeps layers 0, 1 always Full, then every freq-th layer
            # starting from layer 1: {0, 1, 1+freq, 1+2*freq, ...}
            step = max(1, int(round(1.0 / ratio)))
            self.full_layers = {0, 1} | set(range(1, num_layers, step))
        else:
            # All layers are Full = IndexCache disabled
            self.full_layers = set(range(num_layers))

        # Layers 0 and 1 must always be Full.
        # Layer 0: NextN/dense attention, has no preceding Full layer to
        # reuse from. Layer 1: first DSA layer, must run its own indexer
        # to establish the initial index K cache (reference enforces this
        # via max(layer_id-1, 0) which makes both 0 and 1 always Full).
        for protected_layer in (0, 1):
            if protected_layer < num_layers and protected_layer not in self.full_layers:
                logger.warning(
                    f"IndexCache: layer {protected_layer} was not marked as Full. "
                    f"Forcing to Full (required for correctness)."
                )
                self.full_layers.add(protected_layer)

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
