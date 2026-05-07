import json
import argparse
import subprocess
from pathlib import Path

import torch
from PIL import Image
import open_clip
from tqdm import tqdm


def collect_images_recursive(src_dir):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return sorted(
        p for p in Path(src_dir).rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    )

def compute_fid(gen_dir, ref_dir, device="cuda"):
    cmd = [
        "python", "-m", "pytorch_fid",
        "--device", device,
        "--dims", "2048",
        "--num-workers", "0",
        gen_dir,
        ref_dir,
    ]

    print("Running FID:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError("FID failed")

    for line in result.stdout.splitlines():
        if "FID:" in line:
            fid = float(line.split("FID:")[-1].strip())
            return fid

    raise RuntimeError("Failed to parse FID output")

@torch.no_grad()
def compute_clip_score(gen_dir, subset_json, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-g-14",
        pretrained="laion2b_s34b_b88k",
        device=device,
    )
    tokenizer = open_clip.get_tokenizer("ViT-g-14")
    model.eval()

    # prompt mapping
    with open(subset_json, "r") as f:
        subset = json.load(f)

    stem_to_prompt = {
        Path(x["file_name"]).stem: x["prompt"]
        for x in subset
    }

    image_paths = collect_images_recursive(gen_dir)

    pairs = []
    for p in image_paths:
        stem = p.stem
        if stem in stem_to_prompt:
            pairs.append((p, stem_to_prompt[stem]))

    scores = []

    batch_size = 64
    for i in tqdm(range(0, len(pairs), batch_size)):
        batch = pairs[i:i+batch_size]

        imgs = []
        texts = []

        for p, prompt in batch:
            img = Image.open(p).convert("RGB")
            imgs.append(preprocess(img))
            texts.append(prompt)

        imgs = torch.stack(imgs).to(device)
        tokens = tokenizer(texts).to(device)

        img_feat = model.encode_image(imgs)
        txt_feat = model.encode_text(tokens)

        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        sim = (img_feat * txt_feat).sum(dim=-1)
        scores.extend(sim.cpu().tolist())

    return sum(scores) / len(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir", required=True)
    parser.add_argument("--ref_dir", required=True)
    parser.add_argument("--subset_json", required=True)
    parser.add_argument("--out_path", default=None)
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    print("=== Computing FID ===")
    fid = compute_fid(args.gen_dir, args.ref_dir, args.device)

    print("=== Computing CLIP ===")
    clip_score = compute_clip_score(args.gen_dir, args.subset_json, args.device)

    if args.out_path is None:
        out_path = Path(args.gen_dir) / "results.txt"
    else:
        out_path = Path(args.out_path)

    with open(out_path, "w") as f:
        f.write(f"FID: {fid:.6f}\n")
        f.write(f"CLIP: {clip_score:.6f}\n")

    print("\n===== RESULT =====")
    print(f"FID: {fid:.6f}")
    print(f"CLIP: {clip_score:.6f}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()