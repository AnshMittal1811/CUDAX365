import torch
from flash_attn import flash_attn_qkvpacked_func


def main():
    b, s, h, d = 2, 128, 8, 64
    qkv = torch.randn(b, s, 3, h, d, device="cuda", dtype=torch.float16)
    out = flash_attn_qkvpacked_func(qkv, dropout_p=0.0, causal=False)
    torch.cuda.synchronize()
    print(out.shape)


if __name__ == "__main__":
    main()
