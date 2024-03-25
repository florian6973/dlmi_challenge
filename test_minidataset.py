

# test standard sklearn with batch

from dlmi.data.MiniDataset import MiniDataset, Sampler

from torch.utils.data import DataLoader

print("OK")

minidataset = MiniDataset("data/raw")
print(minidataset.classes)

s = Sampler(minidataset.classes, class_per_batch=1, batch_size=4)
print(list(s))
print(list(s))
print(len(list(s)))

dl = DataLoader(minidataset, batch_sampler=s)

for batch in dl:
    print(batch)
    break

