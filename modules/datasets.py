import os
import json
from PIL import Image
from datasets import load_dataset

import torch


class COCODataset(torch.utils.data.Dataset):
    def __init__(self, subset_json: str):
        with open(subset_json, "r") as f:
            self.items = json.load(f)

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
    