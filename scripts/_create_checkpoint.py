"""One-time script to create a checkpoint from crash logs."""
import json
from pathlib import Path

shared_layers = [3, 4, 6, 9, 11, 14, 18, 24, 26, 29, 30, 38, 40, 42, 47, 49, 52]

ckpt = {
    "shared_layers": sorted(shared_layers),
    "step": 3,
    "block_idx": 1,  # completed block 1, crashed in block 2
}

ckpt_path = Path("/cache/.index_cache_dsv3.2_r0.5_checkpoint.json")
with open(ckpt_path, "w") as f:
    json.dump(ckpt, f, indent=2)

print(f"Checkpoint written to {ckpt_path}")
print(f"Shared layers ({len(shared_layers)}): {sorted(shared_layers)}")
print(f"Remaining Full: {61 - len(shared_layers)}")
print(f"Target Full: 31 (need to remove {61 - len(shared_layers) - 31} more)")
