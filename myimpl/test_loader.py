from einops import repeat
import torch
a = torch.arange(6).reshape(2, 3)
print(repeat(a, "i j -> (i n) j", n=2))
print(repeat(a, "i j -> (n i) j", n=2))

