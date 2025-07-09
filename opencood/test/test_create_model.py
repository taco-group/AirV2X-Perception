import torch


def regroup(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    return split_x


x = torch.randn(2, 2, 2)
print(x)
record_len = torch.tensor([0, 2])
y = regroup(x, record_len)
for v in y:
    print(v.shape)

print(y[0])

print(y[1])
