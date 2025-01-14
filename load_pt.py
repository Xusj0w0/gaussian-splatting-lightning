import torch

path = "tmp/partitions/building/partitions/partitions.pt"
data = torch.load(path)
data["extra_data"].update({"dataset_path": "datasets/MegaNeRF/building/colmap"})
torch.save(data, path)
