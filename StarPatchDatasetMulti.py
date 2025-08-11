import os
import glob
from torch.utils.data import Dataset
from StarPatchDataset import StarPatchDataset

class StarPatchDatasetMulti(Dataset):
    def __init__(self, folder1_path, folder2_path=None, patch_size=32, num_negative=10, transform=None):
        folder1_path = os.path.abspath(folder1_path)
        if folder2_path:
            folder2_path = os.path.abspath(folder2_path)

        self.datasets = []

        # STAR PATCHLERİ
        img_files = sorted(glob.glob(os.path.join(folder1_path, "image_*.png")))

        for img_path in img_files:
            idx = os.path.splitext(os.path.basename(img_path))[0].split('_')[1]
            csv_values = [
                os.path.join(folder1_path, f"coords_{idx}.csv"),
                os.path.join(folder1_path, f"coords_{idx.zfill(3)}.csv")
            ]
            csv_path = next((p for p in csv_values if os.path.exists(p)), None)

            if csv_path:
                ds = StarPatchDataset(
                    img_path=img_path,
                    unrelated_path=None,
                    coord_csv=csv_path,
                    patch_size=patch_size,
                    num_negative=num_negative,
                    transform=transform,
                    use_unrelated=False
                )
                if len(ds) > 0:
                    self.datasets.append(ds)

        # UNRELATED PATCHLER
        if folder2_path:
            unrelated_files = sorted(glob.glob(os.path.join(folder2_path, "image_*.png")))
            for unrelated_path in unrelated_files:
                ds = StarPatchDataset(
                    img_path=None,
                    unrelated_path=unrelated_path,
                    coord_csv=None,
                    patch_size=patch_size,
                    num_negative=num_negative // 2,
                    transform=transform,
                    use_unrelated=True
                )
                if len(ds) > 0:
                    self.datasets.append(ds)

        self.lengths = [len(ds) for ds in self.datasets]
        self.total_len = sum(self.lengths)

        if self.total_len == 0:
            raise RuntimeError(f"No patches found in {folder1_path} or {folder2_path}")

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        for ds_len, ds in zip(self.lengths, self.datasets):
            if idx < ds_len:
                return ds[idx]
            idx -= ds_len
