

import os, random, math, shutil
import numpy as np
import pandas as pd
import cv2


N_IMAGES   = 100         
PATCH_SIZE = 512          
OUT_ROOT   = "dataset"    
SPLIT      = (0.70, 0.15, 0.15)  
SEED       = 42
random.seed(SEED); np.random.seed(SEED)


TMP_DIR = "_tmp_starimages"
os.makedirs(TMP_DIR, exist_ok=True)

def add_star(img, cx, cy, sigma):
    y, x = np.ogrid[:PATCH_SIZE, :PATCH_SIZE]
    g = np.exp(-((x-cx)**2 + (y-cy)**2) / (2*sigma**2))
    img += g * 255

def build_one(idx):
    img = np.zeros((PATCH_SIZE, PATCH_SIZE), np.float32)
    n_stars = random.randint(20, 40)
    rows = []
    for _ in range(n_stars):
        cx = random.randint(6, PATCH_SIZE-6)
        cy = random.randint(6, PATCH_SIZE-6)
        sigma = random.uniform(1.0, 2.8)
        add_star(img, cx, cy, sigma)
        rows.append([1, cx/PATCH_SIZE, cy/PATCH_SIZE, 0, 0, round(sigma,3)])


    img += np.random.poisson(lam=2, size=img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)

    name_png = f"image_{idx:03d}.png"
    name_csv = f"coords_{idx:03d}.csv"
    cv2.imwrite(os.path.join(TMP_DIR, name_png), img)
    pd.DataFrame(rows).to_csv(os.path.join(TMP_DIR, name_csv),
                              header=False, index=False)

print("▶ Sentetik yıldız görüntüleri üretiliyor…")
for i in range(N_IMAGES):
    build_one(i)
print("✔ Üretim tamam.")


png_files = sorted([f for f in os.listdir(TMP_DIR) if f.endswith(".png")])
random.shuffle(png_files)

n = len(png_files)
n_train = int(n * SPLIT[0])
n_val   = int(n * SPLIT[1])

splits = {
    "train": png_files[:n_train],
    "val":   png_files[n_train:n_train + n_val],
    "test":  png_files[n_train + n_val:]
}

def mkdir(path):
    os.makedirs(path, exist_ok=True)

for split_name, files in splits.items():
    split_dir = os.path.join(OUT_ROOT, split_name)
    mkdir(split_dir)
    for png in files:
        csv = png.replace("image_", "coords_").replace(".png", ".csv")

    
        shutil.move(os.path.join(TMP_DIR, png),
                    os.path.join(split_dir, png))

        shutil.move(os.path.join(TMP_DIR, csv),
                    os.path.join(split_dir, csv))


shutil.rmtree(TMP_DIR)


def check_csv(folder):
    for csv in os.listdir(folder):
        if not csv.endswith(".csv"): continue
        df = pd.read_csv(os.path.join(folder, csv), header=None)
        assert df.shape[1] == 6, f"{csv} sütun sayısı ≠ 6"
        assert (df.iloc[:,0] == 1).all(), f"{csv} class ≠ 1"
        assert ((df.iloc[:,1:3] >= 0) & (df.iloc[:,1:3] <= 1)).all().all(), f"{csv} x/y out-of-range"

for split in ["train","val","test"]:
    check_csv(os.path.join(OUT_ROOT, split))

print("\n🎉 Veri seti hazır!")
for k,v in splits.items():
    print(f"  {k:<5}: {len(v)} görüntü")

print(f"\nKlasör: {OUT_ROOT}/[train|val|test]  → image_###.png + coords_###.csv")
