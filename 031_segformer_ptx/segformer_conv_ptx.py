import os
import re
from pathlib import Path
import torch
from transformers import SegformerForSemanticSegmentation

os.environ["TORCH_LOGS"] = "output_code"
os.environ["TORCH_COMPILE_DEBUG"] = "1"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "./torchinductor_cache"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    conv = model.segformer.encoder.patch_embeddings[0].proj
    conv = torch.compile(conv, fullgraph=True)
    x = torch.randn(1, 3, 512, 512, device=device)
    _ = conv(x)
    torch.cuda.synchronize()

    cache = Path("./torchinductor_cache")
    ptx = list(cache.rglob("*.ptx"))
    print("ptx files", len(ptx))
    for p in ptx[:5]:
        print(p)

    # print a hint where conv kernel might be
    for p in ptx:
        if re.search(r"conv", p.name):
            print("conv-related ptx", p)
            break


if __name__ == "__main__":
    main()
