import os, glob
from torch.utils.data import Dataset
from StarBoxDataset import StarBoxDataset

class StarBoxDatasetMulti(Dataset):
    def __init__(self, folder_path, patch_size=32, num_negative=10):
        self.datasets = []
        self.total_len = 0
        self.lengths = []
        folder_path = os.path.abspath(folder_path)
        img_files = sorted(glob.glob(os.path.join(folder_path, "image_*.png")))

        for img in img_files:
            idx = os.path.basename(img).split('_')[1].split('.')[0]
            csv = next((os.path.join(folder_path, f) for f in [f"coords_{idx}.csv", f"coords_{idx.zfill(3)}.csv"] if os.path.exists(os.path.join(folder_path, f))), None)
            if csv:
                ds = StarBoxDataset(img, csv, patch_size, num_negative)
                if len(ds) > 0:
                    self.datasets.append(ds)
                    self.lengths.append(len(ds))
                    self.total_len += len(ds)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        for l, ds in zip(self.lengths, self.datasets):
            if idx < l:
                return ds[idx]
            idx -= l
