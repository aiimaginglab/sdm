import os
import json
from PIL import Image
from datasets import load_dataset

import torch
# from torchvision import transforms


class COCODataset(torch.utils.data.Dataset):
    def __init__(self, subset_json: str):
        with open(subset_json, "r") as f:
            self.items = json.load(f)

        # file_name 기준 정렬하면 재현성 관리가 쉬움
        self.items = sorted(self.items, key=lambda x: x["file_name"])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        prompt = item["prompt"]
        if not isinstance(prompt, str):
            prompt = str(prompt)

        return {
            "file_name": item["file_name"],
            "prompt": prompt,
            "negative_prompt": "",
        }

# class COCODataset(torch.utils.data.Dataset):
#     def __init__(self, root_dir, image_size=512, mode="validation"):
#         self.root_dir = root_dir
#         self.image_size = image_size
#         self.dataset = load_dataset("yerevann/coco-karpathy", split=mode)

#         self.transform = transforms.Compose([
#             transforms.Resize(self.image_size),
#             transforms.CenterCrop(self.image_size),
#             transforms.ToTensor(),
#         ])

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         item = self.dataset[idx]

#         filepath = item["filepath"]     # ex: "val2014"
#         filename = item["filename"]     # ex: "COCO_val2014_000000522418.jpg"

#         img_path = os.path.join(self.root_dir, filepath, filename)
#         image = Image.open(img_path).convert("RGB")
#         image = self.transform(image)

#         return {
#             "prompt": item["sentences"][0].strip(),
#             "negative_prompt": item["negative_prompt"].strip() if "negative_prompt" in item else "",
#             "image": image,
#             "filename": item["filename"],
#             "filepath": item["filepath"],
#             "imgid": item["imgid"],
#             "cocoid": item["cocoid"],
#         }