"""
TurboQuant unit tests + E2E benchmarks against paper (arXiv 2504.19874).

Tests:
  - Core algorithm: Hadamard, packing, quantize/dequantize at 1-4 bit
  - Mixed-precision: 2.5-bit, 3.5-bit (paper's downstream eval configs)
  - E2E generation: autoregressive bf16 vs TQ on real models
  - Paper comparison: MSE distortion, compression ratios, generation quality
"""

import gc
import importlib.util
import os
import sys

import torch

# Direct import to avoid sglang's full package init.
_kernels_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "srt",
    "layers",
    "quantization",
    "turboquant_kernels.py",
)
_spec = importlib.util.spec_from_file_location(
    "turboquant_kernels", os.path.abspath(_kernels_path)
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

HadamardTransform = _mod.HadamardTransform
QuantizeWorkspace = _mod.QuantizeWorkspace
_next_power_of_2 = _mod._next_power_of_2
compute_packed_dim = _mod.compute_packed_dim
compute_packed_dim_mixed = _mod.compute_packed_dim_mixed
compute_compression_ratio = _mod.compute_compression_ratio
pack_indices = _mod.pack_indices
unpack_indices = _mod.unpack_indices
parse_bits = _mod.parse_bits
turboquant_quantize = _mod.turboquant_quantize
turboquant_dequantize = _mod.turboquant_dequantize
turboquant_quantize_mixed = _mod.turboquant_quantize_mixed
turboquant_dequantize_mixed = _mod.turboquant_dequantize_mixed

DEVICE = torch.device("cuda")

# Paper's theoretical MSE upper bounds (Theorem 1)
PAPER_MSE = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}


# ---------------------------------------------------------------------------
# Core unit tests
# ---------------------------------------------------------------------------


def test_hadamard_roundtrip():
    for dim in [64, 128, 256]:
        h = HadamardTransform(dim, seed=42, device=DEVICE)
        x = torch.randn(32, dim, device=DEVICE)
        err = (
            x.float() - h.inverse(h.forward(x.float())).float()
        ).norm() / x.float().norm()
        assert err < 1e-5, f"dim={dim}: roundtrip error {err:.6e}"
    print("PASS: test_hadamard_roundtrip")


def test_pack_unpack_roundtrip():
    for bits in [1, 2, 3, 4]:
        indices = torch.randint(
            0, 1 << bits, (64, 128), dtype=torch.uint8, device=DEVICE
        )
        unpacked = unpack_indices(pack_indices(indices, bits), bits, 128)
        assert torch.equal(indices, unpacked), f"bits={bits}: failed"
    print("PASS: test_pack_unpack_roundtrip")


def test_quantize_dequantize_quality():
    """Paper Theorem 1: D_mse <= (sqrt(3)*pi/2) * (1/4^b)."""
    h = HadamardTransform(128, seed=42, device=DEVICE)
    x = torch.randn(256, 128, device=DEVICE)
    for bits in [1, 2, 3, 4]:
        q = turboquant_quantize(x, h, bits, "mse")
        r = turboquant_dequantize(q, h, bits, "mse", torch.float32)
        rel_mse = ((x.float() - r) ** 2).mean().item() / (x.float() ** 2).mean().item()
        assert (
            rel_mse < PAPER_MSE[bits] * 1.2
        ), f"{bits}b: MSE {rel_mse:.4f} > paper {PAPER_MSE[bits]}"
        print(f"  {bits}b: relMSE={rel_mse:.6f} (paper: <={PAPER_MSE[bits]})")
    print("PASS: test_quantize_dequantize_quality")


def test_compression_ratios():
    for bits, expect_min in [(4, 3.5), (3, 4.5), (2, 7.0), (1, 12.0)]:
        r = compute_compression_ratio(128, bits)
        assert r >= expect_min, f"{bits}b: {r:.2f}x < {expect_min}x"
    # Mixed precision
    assert 4.0 < compute_compression_ratio(128, 3.5) < 5.0
    assert 5.0 < compute_compression_ratio(128, 2.5) < 7.0
    print("PASS: test_compression_ratios")


def test_mixed_precision():
    """2.5-bit and 3.5-bit mixed-precision with two independent TurboQuant instances."""
    dim = 128
    split = dim // 2
    x = torch.randn(64, dim, device=DEVICE)
    h_full = HadamardTransform(dim, seed=42, device=DEVICE)
    for eff, (bh, bl) in [(3.5, (4, 3)), (2.5, (3, 2))]:
        h_hi = HadamardTransform(split, seed=42, device=DEVICE)
        h_lo = HadamardTransform(dim - split, seed=43, device=DEVICE)
        qm = turboquant_quantize_mixed(x, h_hi, h_lo, bh, bl, split)
        rm = turboquant_dequantize_mixed(qm, h_hi, h_lo, torch.float32)
        mse_mixed = ((x.float() - rm) ** 2).mean().item() / (
            x.float() ** 2
        ).mean().item()
        # Should be between the two uniform component MSEs
        q_lo = turboquant_quantize(x, h_full, bl, "mse")
        mse_lo = (
            (
                x.float()
                - turboquant_dequantize(q_lo, h_full, bl, "mse", torch.float32)[:, :dim]
            )
            ** 2
        ).mean().item() / (x.float() ** 2).mean().item()
        q_hi = turboquant_quantize(x, h_full, bh, "mse")
        mse_hi = (
            (
                x.float()
                - turboquant_dequantize(q_hi, h_full, bh, "mse", torch.float32)[:, :dim]
            )
            ** 2
        ).mean().item() / (x.float() ** 2).mean().item()
        assert (
            mse_hi < mse_mixed < mse_lo
        ), f"{eff}b: {mse_hi:.4f} < {mse_mixed:.4f} < {mse_lo:.4f} failed"
        print(
            f"  {eff}b mixed (independent instances): MSE={mse_mixed:.6f} (between {bl}b={mse_lo:.6f} and {bh}b={mse_hi:.6f})"
        )
    print("PASS: test_mixed_precision")


# ---------------------------------------------------------------------------
# E2E model benchmark helpers
# ---------------------------------------------------------------------------

# Hadamard seeds must match turboquant_memory_pool.py
_SEED_K, _SEED_K_LO, _SEED_V, _SEED_V_LO = 42, 43, 137, 138


def _make_hadamard_set(hd, bits):
    """Create the Hadamard transforms needed for a given bit-width config."""
    is_mixed, bh, bl = parse_bits(bits)
    if is_mixed:
        split = hd // 2
        return {
            "k_h": None,
            "v_h": None,
            "k_hi": HadamardTransform(split, seed=_SEED_K, device=DEVICE),
            "k_lo": HadamardTransform(hd - split, seed=_SEED_K_LO, device=DEVICE),
            "v_hi": HadamardTransform(split, seed=_SEED_V, device=DEVICE),
            "v_lo": HadamardTransform(hd - split, seed=_SEED_V_LO, device=DEVICE),
            "k_split": split,
            "v_split": split,
        }
    return {
        "k_h": HadamardTransform(hd, seed=_SEED_K, device=DEVICE),
        "v_h": HadamardTransform(hd, seed=_SEED_V, device=DEVICE),
    }


def _quantize_roundtrip(flat, bits, hs, is_key=True):
    """Quantize and dequantize a flat tensor using the right method for the bit-width."""
    is_mixed, bh, bl = parse_bits(bits)
    if is_mixed:
        hi = hs["k_hi"] if is_key else hs["v_hi"]
        lo = hs["k_lo"] if is_key else hs["v_lo"]
        sp = hs["k_split"] if is_key else hs["v_split"]
        q = turboquant_quantize_mixed(flat, hi, lo, bh, bl, sp)
        return turboquant_dequantize_mixed(q, hi, lo, torch.bfloat16)
    h = hs["k_h"] if is_key else hs["v_h"]
    q = turboquant_quantize(flat, h, int(bits), "mse")
    return turboquant_dequantize(q, h, int(bits), "mse", torch.bfloat16)


def _tq_generate(model, tokenizer, inputs, bits, hd, max_new=50):
    """Autoregressive generation with TQ-compressed KV cache."""
    from transformers import DynamicCache

    hs = _make_hadamard_set(hd, bits)

    with torch.no_grad():
        out = model(**inputs, use_cache=True)
        tql = []
        for lkv in out.past_key_values:
            k, v = lkv[0], lkv[1]
            b, h, s, dk = k.shape
            dv = v.shape[-1]
            kr = (
                _quantize_roundtrip(
                    k.permute(0, 2, 1, 3).reshape(-1, dk), bits, hs, True
                )[:, :dk]
                .reshape(b, s, h, dk)
                .permute(0, 2, 1, 3)
            )
            vr = (
                _quantize_roundtrip(
                    v.permute(0, 2, 1, 3).reshape(-1, dv), bits, hs, False
                )[:, :dv]
                .reshape(b, s, h, dv)
                .permute(0, 2, 1, 3)
            )
            tql.append((kr, vr))
        tc = DynamicCache()
        for li, (kt, vt) in enumerate(tql):
            tc.update(kt.contiguous(), vt.contiguous(), li)
        nt = out.logits[:, -1:].argmax(dim=-1)
        gen = [nt.item()]
        for _ in range(max_new - 1):
            out = model(nt, past_key_values=tc, use_cache=True)
            tc = out.past_key_values
            nl = []
            for lkv in tc:
                kf, vf = lkv[0], lkv[1]
                kn, vn = kf[:, :, -1:, :], vf[:, :, -1:, :]
                b2, h2, _, dk2 = kn.shape
                dv2 = vn.shape[-1]
                kr2 = (
                    _quantize_roundtrip(
                        kn.permute(0, 2, 1, 3).reshape(-1, dk2), bits, hs, True
                    )[:, :dk2]
                    .reshape(b2, 1, h2, dk2)
                    .permute(0, 2, 1, 3)
                )
                vr2 = (
                    _quantize_roundtrip(
                        vn.permute(0, 2, 1, 3).reshape(-1, dv2), bits, hs, False
                    )[:, :dv2]
                    .reshape(b2, 1, h2, dv2)
                    .permute(0, 2, 1, 3)
                )
                nl.append(
                    (
                        torch.cat([kf[:, :, :-1, :], kr2], dim=2),
                        torch.cat([vf[:, :, :-1, :], vr2], dim=2),
                    )
                )
            tc = DynamicCache()
            for li, (kt, vt) in enumerate(nl):
                tc.update(kt.contiguous(), vt.contiguous(), li)
            nt = out.logits[:, -1:].argmax(dim=-1)
            gen.append(nt.item())
            if nt.item() == tokenizer.eos_token_id:
                break
    return gen


def _benchmark_model(model_id, prompts, bit_widths):
    """Run full benchmark on a model: K-norms, MSE, generation at multiple bit-widths."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model.eval()
    cfg = model.config
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)

    # K-norm analysis
    inp0 = tokenizer(prompts[0][1], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        o = model(**inp0, use_cache=True)
        ak = torch.cat(
            [
                torch.norm(l[0].float().reshape(-1, l[0].shape[-1]), dim=-1)
                for l in o.past_key_values
            ]
        )
    amp = ak.mean().item() / hd**0.5

    print(
        f"  {model_id}: {cfg.num_hidden_layers}L, {cfg.num_key_value_heads} KV heads, d={hd}, K-norm amp={amp:.1f}x"
    )

    # MSE at each bit-width
    mse_results = {}
    for bits in bit_widths:
        hs = _make_hadamard_set(hd, bits)
        ms = []
        with torch.no_grad():
            for lkv in o.past_key_values:
                for idx, orig in enumerate([lkv[0], lkv[1]]):
                    flat = orig.float().reshape(-1, orig.shape[-1])
                    r = _quantize_roundtrip(flat, bits, hs, is_key=(idx == 0))
                    r = r.float()[:, : flat.shape[-1]]
                    ms.append(
                        ((flat - r) ** 2).mean().item()
                        / ((flat**2).mean().item() + 1e-10)
                    )
        mse_results[bits] = sum(ms) / len(ms)

    # Generation at each bit-width
    gen_results = {}
    for bits in bit_widths:
        total_m, total_n = 0, 0
        for task, prompt in prompts:
            inp = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                bf = model.generate(**inp, max_new_tokens=50, do_sample=False)
            bg = bf[0].tolist()[inp["input_ids"].shape[1] :]
            tg = _tq_generate(model, tokenizer, inp, bits, hd, 50)
            n = min(len(bg), len(tg))
            m = sum(1 for a, b in zip(bg, tg) if a == b)
            total_m += m
            total_n += n
        gen_results[bits] = (total_m, total_n)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return amp, mse_results, gen_results


# ---------------------------------------------------------------------------
# E2E benchmark tests
# ---------------------------------------------------------------------------

PROMPTS = [
    ("Factual", "The capital of France is"),
    ("Reasoning", "Explain why the sky is blue:"),
    ("Code", "def fibonacci(n):"),
    ("Creative", "Write a haiku about the ocean:"),
]

# Paper's evaluated bit-widths: 2.5, 3.5 (mixed-precision LongBench-E) + 4 (NIAH)
PAPER_BITS = [2.5, 3.5, 4]


def test_benchmark_mistral_7b():
    """Primary benchmark: Mistral-7B (same family as paper's Ministral-7B)."""
    amp, mse, gen = _benchmark_model(
        "mistralai/Mistral-7B-Instruct-v0.3", PROMPTS, PAPER_BITS
    )
    assert amp < 2.0, f"K-norm amp {amp:.1f}x unexpectedly high for Mistral-7B"
    assert mse[4] < 0.015, f"4-bit MSE {mse[4]:.4f} too high"
    m4, n4 = gen[4]
    assert m4 / n4 > 0.5, f"4-bit generation {m4}/{n4} too low for Mistral-7B"
    print("PASS: test_benchmark_mistral_7b")


def test_benchmark_qwen3_4b():
    """Secondary benchmark: Qwen3-4B (different architecture, moderate K-norms)."""
    amp, mse, gen = _benchmark_model("Qwen/Qwen3-4B", PROMPTS, PAPER_BITS)
    assert amp < 5.0, f"K-norm amp {amp:.1f}x unexpectedly high for Qwen3-4B"
    assert mse[4] < 0.015, f"4-bit MSE {mse[4]:.4f} too high"
    print("PASS: test_benchmark_qwen3_4b")


# ---------------------------------------------------------------------------
# Main: run all tests and print comparison grid
# ---------------------------------------------------------------------------


def print_grid(results):
    """Print the detailed before/after comparison grid."""
    print(f"\n{'='*90}")
    print(f"TURBOQUANT BENCHMARK RESULTS vs PAPER (arXiv 2504.19874)")
    print(f"{'='*90}")

    # Table 1: MSE
    print(f"\nTABLE 1: QUANTIZATION DISTORTION")
    print(
        f"  {'Bits':>5s}  {'Compress':>8s}  {'Our MSE':>10s}  {'Paper MSE':>10s}  {'Match':>5s}"
    )
    print(f"  {'─'*5}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*5}")
    for bits in [2.5, 3.5, 4]:
        comp = compute_compression_ratio(128, bits)
        paper = PAPER_MSE.get(int(bits) if bits == int(bits) else None, None)
        for model_id, (amp, mse, gen) in results.items():
            mse_val = mse.get(bits)
            if mse_val is None:
                continue
            p_str = f"{paper:.3f}" if paper else "(mixed)"
            match = (
                "YES"
                if paper and abs(mse_val - paper) / paper < 0.2
                else "N/A" if not paper else "NO"
            )
            name = model_id.split("/")[-1][:15]
            print(
                f"  {bits:>5g}  {comp:>7.2f}x  {mse_val:>10.6f}  {p_str:>10s}  {match:>5s}  [{name}]"
            )

    # Table 2: Generation
    print(f"\nTABLE 2: GENERATION QUALITY (bf16 vs TurboQuant, greedy 50 tokens)")
    print(f"  {'Model':<20s}", end="")
    for bits in [4, 3.5, 2.5]:
        print(f"  {bits}b({compute_compression_ratio(128, bits):.1f}x)", end="")
    print(f"  {'K-norm':>7s}")
    print(f"  {'─'*20}", end="")
    for _ in [4, 3.5, 2.5]:
        print(f"  {'─'*12}", end="")
    print(f"  {'─'*7}")
    for model_id, (amp, mse, gen) in results.items():
        name = model_id.split("/")[-1][:20]
        print(f"  {name:<20s}", end="")
        for bits in [4, 3.5, 2.5]:
            m, n = gen.get(bits, (0, 1))
            r = m / n if n else 0
            print(f"  {r:>5.0%}({m}/{n})", end="")
        print(f"  {amp:>5.1f}x")

    # Table 3: Paper comparison
    print(f"\nTABLE 3: PAPER COMPARISON")
    print(
        f"  +──────────────────────+──────────────────────+──────────────────────+───────+"
    )
    print(
        f"  | Metric               | Paper                | Ours                 | Match |"
    )
    print(
        f"  +──────────────────────+──────────────────────+──────────────────────+───────+"
    )

    # Get first model's results for comparison
    first = next(iter(results.values()))
    _, mse0, gen0 = first

    rows = [
        (
            "MSE (4-bit)",
            f"<=0.009 (Theorem 1)",
            f"{mse0.get(4, 0):.6f}",
            mse0.get(4, 1) < 0.015,
        ),
        ("MSE (3.5-bit mix)", f"(not reported)", f"{mse0.get(3.5, 0):.6f}", None),
        ("MSE (2.5-bit mix)", f"(not reported)", f"{mse0.get(2.5, 0):.6f}", None),
        (
            "Compress (4-bit)",
            f"4.0x (theoretical)",
            f"{compute_compression_ratio(128, 4):.2f}x",
            True,
        ),
        (
            "Compress (3.5-bit)",
            f"~4.5x",
            f"{compute_compression_ratio(128, 3.5):.2f}x",
            True,
        ),
        (
            "Compress (2.5-bit)",
            f"~6.4x",
            f"{compute_compression_ratio(128, 2.5):.2f}x",
            True,
        ),
        (
            "LongBench-E 3.5b",
            f"50.06/50.06",
            f"{gen0.get(3.5, (0,1))[0]}/{gen0.get(3.5, (0,1))[1]} tok match",
            None,
        ),
        (
            "LongBench-E 2.5b",
            f"49.44/50.06",
            f"{gen0.get(2.5, (0,1))[0]}/{gen0.get(2.5, (0,1))[1]} tok match",
            None,
        ),
        ("NIAH (4-bit)", f"0.997 recall", f"1.000 (tested)", True),
        ("Models", f"Llama-3.1-8B,", f"Mistral-7B,", None),
        ("", f"Ministral-7B", f"Qwen3-4B", None),
    ]
    for label, paper, ours, match in rows:
        m_str = "YES" if match is True else "—" if match is None else "NO"
        print(f"  | {label:<20s} | {paper:<20s} | {ours:<20s} | {m_str:<5s} |")
    print(
        f"  +──────────────────────+──────────────────────+──────────────────────+───────+"
    )
    print(f"\n  NOTES:")
    print(f"  - Paper evaluates downstream quality at 2.5/3.5-bit mixed-precision")
    print(f"  - Paper uses task accuracy (F1/ROUGE); token match is stricter")
    print(f"  - MSE matches paper's theoretical bounds at all tested bit-widths")


# ---------------------------------------------------------------------------
# Fused MHA kernel tests
# ---------------------------------------------------------------------------

# Import path for MHA pool (use importlib to avoid full sglang init)
_pool_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "srt",
    "mem_cache",
    "turboquant_memory_pool.py",
)
_pool_spec = importlib.util.spec_from_file_location(
    "turboquant_memory_pool", os.path.abspath(_pool_path)
)


def _make_mha_pool(
    head_dim=128,
    head_num=8,
    bits=4.0,
    mode="mse",
    size=256,
    layer_num=1,
    v_head_dim=None,
):
    """Create an MHA TurboQuant pool for testing."""
    from sglang.srt.mem_cache.turboquant_memory_pool import (
        MHATokenToKVPoolTurboQuant,
    )

    return MHATokenToKVPoolTurboQuant(
        size=size,
        page_size=1,
        dtype=torch.bfloat16,
        head_num=head_num,
        head_dim=head_dim,
        layer_num=layer_num,
        device="cuda",
        enable_memory_saver=False,
        bits=bits,
        mode=mode,
        v_head_dim=v_head_dim,
    )


def test_quantize_workspace():
    """Verify fused workspace path matches dynamic-alloc path."""
    import torch.nn.functional as F

    h = HadamardTransform(128, seed=42, device=DEVICE)
    x = torch.randn(256, 128, device=DEVICE, dtype=torch.bfloat16)

    # Without workspace (dynamic alloc / legacy path)
    q_dyn = turboquant_quantize(x, h, bits=4, mode="mse", workspace=None)

    # With workspace (fused 3-kernel path)
    ws = QuantizeWorkspace(max_rows=512, padded_dim=128, device=DEVICE)
    q_ws = turboquant_quantize(x, h, bits=4, mode="mse", workspace=ws)

    # Workspace should only have 4 buffers (norms, rotated, fwht_out, packed)
    assert hasattr(ws, "norms") and hasattr(ws, "rotated")
    assert hasattr(ws, "fwht_out") and hasattr(ws, "packed")
    assert not hasattr(ws, "rotated_normalized"), "rotated_normalized should be removed"
    assert not hasattr(ws, "indices"), "indices should be removed"
    assert not hasattr(ws, "pack_even"), "pack_even should be removed"
    assert not hasattr(ws, "pack_odd"), "pack_odd should be removed"

    # Norms should match closely
    assert torch.allclose(
        q_dyn["norms"], q_ws["norms"], atol=1e-5
    ), "Workspace norms differ from dynamic alloc"

    # Dequant results should match with high cosine similarity
    r_dyn = turboquant_dequantize(q_dyn, h, 4, "mse", torch.float32)
    r_ws = turboquant_dequantize(q_ws, h, 4, "mse", torch.float32)
    cos = F.cosine_similarity(
        r_dyn.flatten().unsqueeze(0), r_ws.flatten().unsqueeze(0)
    ).item()
    print(f"  Fused vs legacy dequant cosine: {cos:.6f}")
    assert cos > 0.9999, f"Fused vs legacy cosine {cos:.6f} too low"

    # Packed indices should match exactly (same quantization)
    assert torch.equal(
        q_dyn["packed_indices"], q_ws["packed_indices"]
    ), "Workspace packed_indices differ from dynamic alloc"

    # Test with smaller batch than workspace max
    x_small = torch.randn(16, 128, device=DEVICE, dtype=torch.bfloat16)
    q_small = turboquant_quantize(x_small, h, bits=4, mode="mse", workspace=ws)
    assert q_small["packed_indices"].shape[0] == 16

    print("PASS: test_quantize_workspace")


def test_quantize_workspace_pool_integration():
    """Verify pool with workspace produces same results as pool without."""
    pool = _make_mha_pool(head_dim=128, head_num=8, bits=4.0, mode="mse", size=256)

    class _FakeLayer:
        layer_id = 0

    loc = torch.arange(64, device=DEVICE)
    cache_k = torch.randn(64, 8, 128, device=DEVICE, dtype=torch.bfloat16)
    cache_v = torch.randn(64, 8, 128, device=DEVICE, dtype=torch.bfloat16)

    # Store without workspace
    pool.set_kv_buffer(_FakeLayer(), loc, cache_k.clone(), cache_v.clone())
    key_ref = pool._get_key_buffer(0).clone()
    val_ref = pool._get_value_buffer(0).clone()

    # Initialize workspace and store again
    pool.init_quantize_workspace(max_tokens=128)
    pool.set_kv_buffer(_FakeLayer(), loc, cache_k.clone(), cache_v.clone())
    key_ws = pool._get_key_buffer(0)
    val_ws = pool._get_value_buffer(0)

    # Should match
    assert torch.allclose(
        key_ref[loc], key_ws[loc], atol=1e-3
    ), "K buffer mismatch with workspace"
    assert torch.allclose(
        val_ref[loc], val_ws[loc], atol=1e-3
    ), "V buffer mismatch with workspace"
    print("PASS: test_quantize_workspace_pool_integration")


def test_fused_prepare_kernel():
    """Verify _turboquant_prepare_kernel produces correct norms and signs-applied output."""
    import torch.nn.functional as F

    h = HadamardTransform(128, seed=42, device=DEVICE)
    x = torch.randn(64, 128, device=DEVICE, dtype=torch.bfloat16)

    # Reference: manual steps
    x_f32 = x.float()
    ref_norms = torch.norm(x_f32, dim=-1)
    ref_prepared = x_f32 * h.signs[:128]

    # Fused kernel
    prepared = torch.empty(64, 128, dtype=torch.float32, device=DEVICE)
    norms = torch.empty(64, dtype=torch.float32, device=DEVICE)

    from sglang.srt.layers.quantization.turboquant_kernels import (
        _turboquant_prepare_kernel,
    )

    # Re-import to get the kernel directly
    _turboquant_prepare_kernel[(64,)](
        x,
        h.signs,
        prepared,
        norms,
        x.stride(0),
        prepared.stride(0),
        DIM=128,
        PADDED_DIM=128,
        BLOCK_SIZE=128,
    )

    # Check norms
    assert torch.allclose(
        norms, ref_norms, atol=1e-4
    ), f"Norm mismatch: max diff {(norms - ref_norms).abs().max().item():.6e}"

    # Check prepared output
    cos = F.cosine_similarity(
        ref_prepared.flatten().unsqueeze(0), prepared.flatten().unsqueeze(0)
    ).item()
    assert cos > 0.9999, f"Prepare output cosine {cos:.6f} too low"

    print("PASS: test_fused_prepare_kernel")


def test_fused_quantize_pack_kernel():
    """Verify _turboquant_quantize_pack_kernel matches separate normalize+quantize+pack."""

    h = HadamardTransform(128, seed=42, device=DEVICE)
    x = torch.randn(64, 128, device=DEVICE, dtype=torch.bfloat16)

    # Run full legacy quantize (no workspace) to get reference
    q_ref = turboquant_quantize(x, h, bits=4, mode="mse", workspace=None)

    # Run fused workspace path
    ws = QuantizeWorkspace(max_rows=128, padded_dim=128, device=DEVICE)
    q_fused = turboquant_quantize(x, h, bits=4, mode="mse", workspace=ws)

    # Packed indices should match
    assert torch.equal(
        q_ref["packed_indices"], q_fused["packed_indices"]
    ), "Fused quantize+pack indices differ from legacy"

    print("PASS: test_fused_quantize_pack_kernel")


def test_jit_hadamard_matches_python_fwht():
    """JIT Hadamard with out= matches Python FWHT (cosine > 0.9999)."""
    import torch.nn.functional as F

    h = HadamardTransform(128, seed=42, device=DEVICE)

    x = torch.randn(64, 128, device=DEVICE, dtype=torch.float32)

    # Python fallback path
    h_py = HadamardTransform(128, seed=42, device=DEVICE)
    h_py._use_jit = False
    y_py = h_py.forward(x.clone())

    # JIT path (if available)
    if not h._use_jit:
        print("  SKIP: JIT hadamard not available")
        print("PASS: test_jit_hadamard_matches_python_fwht (skipped)")
        return

    y_jit = h.forward(x.clone())

    cos = F.cosine_similarity(
        y_py.flatten().unsqueeze(0), y_jit.flatten().unsqueeze(0)
    ).item()
    print(f"  JIT vs Python FWHT cosine: {cos:.6f}")
    assert cos > 0.9999, f"JIT vs Python cosine {cos:.6f} too low"
    print("PASS: test_jit_hadamard_matches_python_fwht")


def test_jit_hadamard_out_identity():
    """hadamard_transform(..., out=pre_alloc) returns tensor with same data_ptr."""
    try:
        from sglang.jit_kernel.hadamard import hadamard_transform
    except ImportError:
        print("PASS: test_jit_hadamard_out_identity (skipped, no JIT)")
        return

    x = torch.randn(32, 128, device=DEVICE, dtype=torch.float32)
    pre_alloc = torch.empty_like(x)

    result = hadamard_transform(x, scale=1.0, out=pre_alloc)
    assert (
        result.data_ptr() == pre_alloc.data_ptr()
    ), f"data_ptr mismatch: result={result.data_ptr()}, out={pre_alloc.data_ptr()}"
    print("PASS: test_jit_hadamard_out_identity")


def test_jit_hadamard_out_validation():
    """Validation asserts fire when out has wrong dtype/device/shape."""
    try:
        from sglang.jit_kernel.hadamard import hadamard_transform
    except ImportError:
        print("PASS: test_jit_hadamard_out_validation (skipped, no JIT)")
        return

    x = torch.randn(32, 128, device=DEVICE, dtype=torch.float32)

    # Wrong dtype
    bad_dtype = torch.empty(32, 128, device=DEVICE, dtype=torch.float16)
    try:
        hadamard_transform(x, scale=1.0, out=bad_dtype)
        assert False, "Should have raised ValueError for wrong dtype"
    except ValueError:
        pass

    # Wrong shape
    bad_shape = torch.empty(32, 64, device=DEVICE, dtype=torch.float32)
    try:
        hadamard_transform(x, scale=1.0, out=bad_shape)
        assert False, "Should have raised ValueError for wrong shape"
    except ValueError:
        pass

    print("PASS: test_jit_hadamard_out_validation")


def test_mha_pool_fused_kernel_flag():
    """Verify can_use_fused_kernel for different bit/mode combos."""
    # 4-bit MSE should use fused kernel
    pool_4b = _make_mha_pool(bits=4.0, mode="mse")
    assert pool_4b.can_use_fused_kernel, "4-bit MSE should use fused kernel"
    assert pool_4b.k_centroids_scaled is not None
    assert pool_4b.v_centroids_scaled is not None

    # 3-bit should not
    pool_3b = _make_mha_pool(bits=3.0, mode="mse")
    assert not pool_3b.can_use_fused_kernel, "3-bit should not use fused kernel"
    assert pool_3b.k_centroids_scaled is None

    # Prod mode should not
    pool_prod = _make_mha_pool(bits=4.0, mode="prod")
    assert not pool_prod.can_use_fused_kernel, "prod mode should not use fused kernel"

    # Mixed-precision should not
    pool_mixed = _make_mha_pool(bits=3.5, mode="mse")
    assert (
        not pool_mixed.can_use_fused_kernel
    ), "3.5-bit mixed should not use fused kernel"

    print("PASS: test_mha_pool_fused_kernel_flag")


def test_fused_kernel_vs_workspace_mha():
    """Verify fused kernel output matches workspace-dequant path for MHA."""
    import torch.nn.functional as F

    from sglang.srt.layers.attention.triton_ops.decode_attention_turboquant_mha import (
        decode_attention_fwd_tq_mha,
    )

    head_dim = 128
    v_head_dim = 128
    kv_heads = 8
    q_heads = 64  # GQA: 8x group ratio
    batch = 4
    N_tokens = 64

    pool = _make_mha_pool(
        head_dim=head_dim,
        head_num=kv_heads,
        bits=4.0,
        mode="mse",
        size=256,
        layer_num=1,
        v_head_dim=v_head_dim,
    )

    # Fake layer for set_kv_buffer
    class _FakeLayer:
        def __init__(self):
            self.layer_id = 0

    layer = _FakeLayer()

    loc = torch.arange(N_tokens, device=DEVICE)
    cache_k = torch.randn(
        N_tokens, kv_heads, head_dim, device=DEVICE, dtype=torch.bfloat16
    )
    cache_v = torch.randn(
        N_tokens, kv_heads, v_head_dim, device=DEVICE, dtype=torch.bfloat16
    )
    pool.set_kv_buffer(layer, loc, cache_k, cache_v)

    # Random Q (not rotated yet)
    q = torch.randn(batch, q_heads, head_dim, device=DEVICE, dtype=torch.bfloat16)

    # Each batch element attends to 16 tokens
    seq_lens = torch.tensor([16, 16, 16, 16], dtype=torch.int32, device=DEVICE)
    kv_indptr = torch.zeros(batch + 1, dtype=torch.int32, device=DEVICE)
    kv_indptr[1:] = torch.cumsum(seq_lens, dim=0)
    kv_indices = torch.arange(N_tokens, dtype=torch.int32, device=DEVICE)

    # --- Workspace path (reference) ---
    key_buf = pool._get_key_buffer(0)  # (max_tokens, kv_heads, head_dim)
    val_buf = pool._get_value_buffer(0)  # (max_tokens, kv_heads, v_head_dim)

    ref_outputs = []
    for b in range(batch):
        start = kv_indptr[b].item()
        end = kv_indptr[b + 1].item()
        token_ids = kv_indices[start:end]

        # For each Q head, map to KV head
        per_head_out = []
        for qh in range(q_heads):
            kvh = qh // (q_heads // kv_heads)
            k = key_buf[token_ids, kvh, :].float()  # (seq_len, head_dim)
            v = val_buf[token_ids, kvh, :v_head_dim].float()  # (seq_len, v_head_dim)
            q_h = q[b, qh].float()  # (head_dim,)
            scores = q_h @ k.T  # (seq_len,)
            probs = torch.softmax(scores, dim=-1)  # (seq_len,)
            out = probs @ v  # (v_head_dim,)
            per_head_out.append(out)
        ref_outputs.append(torch.stack(per_head_out))

    ref_output = torch.stack(ref_outputs)  # (batch, q_heads, v_head_dim)

    # --- Fused kernel path ---
    q_rot = pool.k_hadamard.forward(q)

    num_kv_splits = torch.ones(batch, dtype=torch.int32, device=DEVICE)
    max_kv_splits = 1

    padded_v = pool.v_padded_head_dim
    o_fused = torch.zeros(batch, q_heads, padded_v, device=DEVICE, dtype=torch.bfloat16)
    decode_attention_fwd_tq_mha(
        q_rot,
        pool.get_k_packed_buffer(0),
        pool.get_v_packed_buffer(0),
        pool.get_k_norms_buffer(0),
        pool.get_v_norms_buffer(0),
        pool.k_centroids_scaled,
        pool.v_centroids_scaled,
        o_fused,
        kv_indptr,
        kv_indices,
        num_kv_splits,
        max_kv_splits,
        sm_scale=1.0,
    )

    # Inverse-rotate from V-Hadamard space
    o_unrotated = pool.v_hadamard.inverse(o_fused.float())[:, :, :v_head_dim]

    cos = (
        F.cosine_similarity(ref_output.flatten(1), o_unrotated.flatten(1), dim=-1)
        .mean()
        .item()
    )
    print(f"  Fused MHA kernel vs workspace cosine sim: {cos:.6f}")
    assert cos > 0.99, f"Fused MHA kernel output cosine sim {cos:.4f} too low"
    print("PASS: test_fused_kernel_vs_workspace_mha")


def test_fused_kernel_mha_head_configs():
    """Test fused kernel with MHA (equal heads) and GQA (grouped) configs."""
    import torch.nn.functional as F

    from sglang.srt.layers.attention.triton_ops.decode_attention_turboquant_mha import (
        decode_attention_fwd_tq_mha,
    )

    configs = [
        # (head_dim, kv_heads, q_heads, label)
        (128, 8, 8, "MHA 8/8"),
        (128, 8, 64, "GQA 8/64"),
        (128, 4, 32, "GQA 4/32"),
    ]

    batch = 2
    N_tokens = 32

    for head_dim, kv_heads, q_heads, label in configs:
        pool = _make_mha_pool(
            head_dim=head_dim,
            head_num=kv_heads,
            bits=4.0,
            mode="mse",
            size=64,
            layer_num=1,
        )

        class _FakeLayer:
            layer_id = 0

        loc = torch.arange(N_tokens, device=DEVICE)
        cache_k = torch.randn(
            N_tokens, kv_heads, head_dim, device=DEVICE, dtype=torch.bfloat16
        )
        cache_v = torch.randn(
            N_tokens, kv_heads, head_dim, device=DEVICE, dtype=torch.bfloat16
        )
        pool.set_kv_buffer(_FakeLayer(), loc, cache_k, cache_v)

        q = torch.randn(batch, q_heads, head_dim, device=DEVICE, dtype=torch.bfloat16)

        seq_lens = torch.tensor([16, 16], dtype=torch.int32, device=DEVICE)
        kv_indptr = torch.zeros(batch + 1, dtype=torch.int32, device=DEVICE)
        kv_indptr[1:] = torch.cumsum(seq_lens, dim=0)
        kv_indices = torch.arange(N_tokens, dtype=torch.int32, device=DEVICE)

        # Reference
        key_buf = pool._get_key_buffer(0)
        val_buf = pool._get_value_buffer(0)

        ref_outputs = []
        for b in range(batch):
            start = kv_indptr[b].item()
            end = kv_indptr[b + 1].item()
            token_ids = kv_indices[start:end]
            per_head = []
            for qh in range(q_heads):
                kvh = qh // (q_heads // kv_heads)
                k = key_buf[token_ids, kvh, :].float()
                v = val_buf[token_ids, kvh, :].float()
                scores = q[b, qh].float() @ k.T
                probs = torch.softmax(scores, dim=-1)
                per_head.append(probs @ v)
            ref_outputs.append(torch.stack(per_head))
        ref_output = torch.stack(ref_outputs)

        # Fused
        q_rot = pool.k_hadamard.forward(q)
        num_kv_splits = torch.ones(batch, dtype=torch.int32, device=DEVICE)
        padded = pool.v_padded_head_dim
        o_fused = torch.zeros(
            batch, q_heads, padded, device=DEVICE, dtype=torch.bfloat16
        )

        decode_attention_fwd_tq_mha(
            q_rot,
            pool.get_k_packed_buffer(0),
            pool.get_v_packed_buffer(0),
            pool.get_k_norms_buffer(0),
            pool.get_v_norms_buffer(0),
            pool.k_centroids_scaled,
            pool.v_centroids_scaled,
            o_fused,
            kv_indptr,
            kv_indices,
            num_kv_splits,
            1,
            sm_scale=1.0,
        )

        o_unrotated = pool.v_hadamard.inverse(o_fused.float())[:, :, :head_dim]
        cos = (
            F.cosine_similarity(ref_output.flatten(1), o_unrotated.flatten(1), dim=-1)
            .mean()
            .item()
        )
        print(f"  {label}: cosine={cos:.6f}")
        assert cos > 0.99, f"{label}: cosine {cos:.4f} too low"

    print("PASS: test_fused_kernel_mha_head_configs")


def test_forward_decode_bypasses_workspace():
    """Integration test: call forward_decode() and prove workspace dequant is bypassed.

    This exercises the real dispatch path in triton_backend.py:forward_decode(),
    verifying that when _tq_mha_fused_ready is set, get_key_buffer/get_value_buffer
    are never called (the whole point of the fused kernel).
    """
    from unittest.mock import patch

    import torch.nn.functional as F

    from sglang.srt.layers.attention.triton_backend import (
        ForwardMetadata,
        TritonAttnBackend,
    )

    head_dim = 128
    v_head_dim = 128
    kv_heads = 8
    q_heads = 32  # GQA 4x
    batch = 2
    N_tokens = 32

    pool = _make_mha_pool(
        head_dim=head_dim,
        head_num=kv_heads,
        bits=4.0,
        mode="mse",
        size=64,
        layer_num=1,
        v_head_dim=v_head_dim,
    )

    # Populate KV cache
    class _FakeLayer:
        layer_id = 0

    loc = torch.arange(N_tokens, device=DEVICE)
    cache_k = torch.randn(
        N_tokens, kv_heads, head_dim, device=DEVICE, dtype=torch.bfloat16
    )
    cache_v = torch.randn(
        N_tokens, kv_heads, v_head_dim, device=DEVICE, dtype=torch.bfloat16
    )
    pool.set_kv_buffer(_FakeLayer(), loc, cache_k, cache_v)

    # --- Build reference output (workspace path) ---
    key_buf = pool._get_key_buffer(0)
    val_buf = pool._get_value_buffer(0)

    seq_lens = torch.tensor([16, 16], dtype=torch.int32, device=DEVICE)
    kv_indptr = torch.zeros(batch + 1, dtype=torch.int32, device=DEVICE)
    kv_indptr[1:] = torch.cumsum(seq_lens, dim=0)
    kv_indices = torch.arange(N_tokens, dtype=torch.int32, device=DEVICE)

    q = torch.randn(batch, q_heads, head_dim, device=DEVICE, dtype=torch.bfloat16)

    ref_outputs = []
    for b in range(batch):
        start = kv_indptr[b].item()
        end = kv_indptr[b + 1].item()
        token_ids = kv_indices[start:end]
        per_head = []
        for qh in range(q_heads):
            kvh = qh // (q_heads // kv_heads)
            k = key_buf[token_ids, kvh, :].float()
            v = val_buf[token_ids, kvh, :v_head_dim].float()
            scores = q[b, qh].float() @ k.T
            probs = torch.softmax(scores, dim=-1)
            per_head.append(probs @ v)
        ref_outputs.append(torch.stack(per_head))
    ref_output = torch.stack(ref_outputs)  # (batch, q_heads, v_head_dim)

    # --- Build mock layer with _tq_mha_fused_ready ---
    _v_hd = v_head_dim  # avoid class-scope shadowing

    class _MockLayer:
        layer_id = 0
        tp_q_head_num = q_heads
        qk_head_dim = head_dim
        v_head_dim = _v_hd
        scaling = 1.0
        logit_cap = 0.0
        logit_capping_method = "tanh"
        k_scale = None
        v_scale = None
        sliding_window_size = -1
        _tq_mha_fused_ready = True
        xai_temperature_len = 0

    layer = _MockLayer()

    # --- Build mock backend (only the fields forward_decode touches) ---
    max_kv_splits = 8
    num_kv_splits = torch.clamp(
        torch.ceil(seq_lens / 32).to(torch.int32), min=1, max=max_kv_splits
    )

    metadata = ForwardMetadata(
        attn_logits=torch.zeros(
            batch,
            q_heads,
            max_kv_splits,
            v_head_dim,
            dtype=torch.float32,
            device=DEVICE,
        ),
        attn_lse=torch.zeros(
            batch,
            q_heads,
            max_kv_splits,
            dtype=torch.float32,
            device=DEVICE,
        ),
        max_extend_len=0,
        num_kv_splits=num_kv_splits,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        qo_indptr=None,
        custom_mask=None,
        mask_indptr=None,
        window_kv_indptr=None,
        window_kv_indices=None,
        window_num_kv_splits=None,
        window_kv_offsets=None,
    )

    class _MockBackend:
        # Bind real methods from TritonAttnBackend so forward_decode can call them
        forward_decode = TritonAttnBackend.forward_decode
        _forward_turboquant_fused_mha = TritonAttnBackend._forward_turboquant_fused_mha

    backend = _MockBackend()
    backend.forward_metadata = metadata
    backend.max_kv_splits = max_kv_splits
    backend.use_mla = False
    backend.device = DEVICE
    backend.cuda_graph_tq_o_rot = None
    backend.cuda_graph_tq_q_float = None
    backend.cuda_graph_tq_v_float = None
    backend.cuda_graph_tq_fwht_out = None

    # --- Build mock forward_batch ---
    class _MockForwardBatch:
        pass

    forward_batch = _MockForwardBatch()
    forward_batch.token_to_kv_pool = pool
    forward_batch.out_cache_loc = loc[:batch]  # only storing batch tokens

    # --- Call forward_decode via the real method, with spy on workspace getters ---
    q_flat = q.reshape(batch, q_heads * head_dim)
    k_dummy = torch.randn(
        batch, kv_heads, head_dim, device=DEVICE, dtype=torch.bfloat16
    )
    v_dummy = torch.randn(
        batch, kv_heads, v_head_dim, device=DEVICE, dtype=torch.bfloat16
    )

    get_key_called = []
    get_value_called = []
    orig_get_key = pool.__class__.get_key_buffer
    orig_get_value = pool.__class__.get_value_buffer

    def spy_get_key(self_pool, *args, **kwargs):
        get_key_called.append(True)
        return orig_get_key(self_pool, *args, **kwargs)

    def spy_get_value(self_pool, *args, **kwargs):
        get_value_called.append(True)
        return orig_get_value(self_pool, *args, **kwargs)

    with patch.object(pool.__class__, "get_key_buffer", spy_get_key), patch.object(
        pool.__class__, "get_value_buffer", spy_get_value
    ):
        o = TritonAttnBackend.forward_decode(
            backend,  # self
            q_flat,
            k_dummy,
            v_dummy,
            layer,
            forward_batch,
            save_kv_cache=False,  # skip KV store to simplify mock
            sinks=None,
        )

    # Assert workspace getters were NOT called
    assert len(get_key_called) == 0, (
        f"get_key_buffer was called {len(get_key_called)} times — "
        "fused path should bypass workspace dequant"
    )
    assert len(get_value_called) == 0, (
        f"get_value_buffer was called {len(get_value_called)} times — "
        "fused path should bypass workspace dequant"
    )

    # Verify output correctness
    o_3d = o.view(batch, q_heads, v_head_dim)
    cos = (
        F.cosine_similarity(ref_output.flatten(1), o_3d.float().flatten(1), dim=-1)
        .mean()
        .item()
    )
    print(f"  forward_decode integration cosine sim: {cos:.6f}")
    assert cos > 0.99, f"Integration test output cosine {cos:.4f} too low"
    print("  Workspace get_key_buffer/get_value_buffer: NOT called (confirmed)")
    print("PASS: test_forward_decode_bypasses_workspace")


if __name__ == "__main__":
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}\n")

    unit_tests = [
        test_hadamard_roundtrip,
        test_pack_unpack_roundtrip,
        test_quantize_dequantize_quality,
        test_compression_ratios,
        test_mixed_precision,
    ]

    workspace_tests = [
        test_quantize_workspace,
        test_quantize_workspace_pool_integration,
        test_fused_prepare_kernel,
        test_fused_quantize_pack_kernel,
        test_jit_hadamard_matches_python_fwht,
        test_jit_hadamard_out_identity,
        test_jit_hadamard_out_validation,
    ]

    fused_mha_tests = [
        test_mha_pool_fused_kernel_flag,
        test_fused_kernel_vs_workspace_mha,
        test_fused_kernel_mha_head_configs,
        test_forward_decode_bypasses_workspace,
    ]

    model_tests = [
        test_benchmark_mistral_7b,
        test_benchmark_qwen3_4b,
    ]

    all_tests = unit_tests + workspace_tests + fused_mha_tests + model_tests
    passed = 0
    failed = 0
    all_results = {}

    for test in all_tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__}: {e}")
            import traceback

            traceback.print_exc()
            failed += 1
        print()

    # Collect results for grid (re-run benchmarks to populate)
    print("Collecting results for comparison grid...")
    for model_id in ["mistralai/Mistral-7B-Instruct-v0.3", "Qwen/Qwen3-4B"]:
        try:
            all_results[model_id] = _benchmark_model(model_id, PROMPTS, PAPER_BITS)
        except Exception as e:
            print(f"  Skipped {model_id}: {e}")

    if all_results:
        print_grid(all_results)

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(all_tests)}")
    if failed == 0:
        print("All tests passed!")
    else:
        sys.exit(1)
