import torch

from pathlib import Path

from typing import List

from ...Data import dirs


def load_parents(
    folder: Path = dirs.self_training_parents,
    eval: bool = True
) -> List[torch.nn.Module]:
    parents = []

    print("Loading Parents")

    for f in folder.iterdir():
        if not f.name.endswith("pt"):
            continue
        model = torch.load(f)
        if eval: model.eval()
        parents.append(model)
        print(f)
        print(model)

    return parents

