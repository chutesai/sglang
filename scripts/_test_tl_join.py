"""Quick test to check tl.join semantics in Triton 3.5.1."""

import torch
import triton
import triton.language as tl


@triton.jit
def test_join_1d(a_ptr, b_ptr, out_ptr, M: tl.constexpr):
    offs = tl.arange(0, M)
    a = tl.load(a_ptr + offs).to(tl.float32)
    b = tl.load(b_ptr + offs).to(tl.float32)
    joined = tl.join(a, b)  # (M, 2)? or (2*M,)?
    flat = tl.reshape(joined, [2 * M])
    offs2 = tl.arange(0, 2 * M)
    tl.store(out_ptr + offs2, flat)


@triton.jit
def test_cat_1d(a_ptr, b_ptr, out_ptr, M: tl.constexpr):
    offs = tl.arange(0, M)
    a = tl.load(a_ptr + offs).to(tl.float32)
    b = tl.load(b_ptr + offs).to(tl.float32)
    catted = tl.cat(a, b)  # should be (2*M,)
    offs2 = tl.arange(0, 2 * M)
    tl.store(out_ptr + offs2, catted)


@triton.jit
def test_join_2d(
    a_ptr,
    b_ptr,
    out_ptr,
    stride_a0,
    stride_b0,
    stride_o0,
    N: tl.constexpr,
    M: tl.constexpr,
):
    """Test tl.join on 2D tensors: a=(N,M), b=(N,M) -> joined shape?"""
    row = tl.program_id(0)
    offs = tl.arange(0, M)
    a = tl.load(a_ptr + row * stride_a0 + offs).to(tl.float32)
    b = tl.load(b_ptr + row * stride_b0 + offs).to(tl.float32)
    joined = tl.join(a, b)  # (M, 2)
    flat = tl.reshape(joined, [2 * M])
    offs2 = tl.arange(0, 2 * M)
    tl.store(out_ptr + row * stride_o0 + offs2, flat)


def main():
    M = 4
    a = torch.arange(M, dtype=torch.float32, device="cuda")  # [0,1,2,3]
    b = torch.arange(10, 10 + M, dtype=torch.float32, device="cuda")  # [10,11,12,13]

    # Test 1D join
    out_join = torch.zeros(2 * M, dtype=torch.float32, device="cuda")
    test_join_1d[(1,)](a, b, out_join, M=M)
    print(f"1D join:  a={a.tolist()}, b={b.tolist()}")
    print(f"  result: {out_join.tolist()}")
    if out_join.tolist() == [0, 10, 1, 11, 2, 12, 3, 13]:
        print("  -> INTERLEAVE (a[0],b[0],a[1],b[1],...)")
    elif out_join.tolist() == [0, 1, 2, 3, 10, 11, 12, 13]:
        print("  -> CONCATENATE (a..., b...)")
    else:
        print("  -> UNKNOWN layout!")

    # Test 1D cat
    out_cat = torch.zeros(2 * M, dtype=torch.float32, device="cuda")
    test_cat_1d[(1,)](a, b, out_cat, M=M)
    print(f"\n1D cat:  a={a.tolist()}, b={b.tolist()}")
    print(f"  result: {out_cat.tolist()}")
    if out_cat.tolist() == [0, 10, 1, 11, 2, 12, 3, 13]:
        print("  -> INTERLEAVE")
    elif out_cat.tolist() == [0, 1, 2, 3, 10, 11, 12, 13]:
        print("  -> CONCATENATE")
    else:
        print("  -> UNKNOWN layout!")

    # Test 2D join
    N = 2
    a2 = torch.arange(N * M, dtype=torch.float32, device="cuda").reshape(N, M)
    b2 = (torch.arange(N * M, dtype=torch.float32, device="cuda") + 100).reshape(N, M)
    out2 = torch.zeros(N, 2 * M, dtype=torch.float32, device="cuda")
    test_join_2d[(N,)](
        a2, b2, out2, a2.stride(0), b2.stride(0), out2.stride(0), N=N, M=M
    )
    print(f"\n2D join:")
    for i in range(N):
        print(f"  row {i}: a={a2[i].tolist()}, b={b2[i].tolist()}")
        print(f"    result: {out2[i].tolist()}")


if __name__ == "__main__":
    main()
