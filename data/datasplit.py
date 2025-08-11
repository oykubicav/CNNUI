import cv2, numpy as np, pandas as pd, os, random, math

N_POS = 200     
N_NEG = 200     
SIZE  = 512    
OUT   = "dataset"  
os.makedirs(f"{OUT}/stars", exist_ok=True)
os.makedirs(f"{OUT}/neg",   exist_ok=True)


def add_star(img, cx, cy, sigma):
    y, x = np.ogrid[:SIZE, :SIZE]
    g = np.exp(-((x-cx)**2 + (y-cy)**2) / (2*sigma**2))
    img += g * 255

def add_blob(img, cx, cy, rx, ry):
    y, x = np.ogrid[:SIZE, :SIZE]
    blob = np.exp(-(((x-cx)/rx)**2 + ((y-cy)/ry)**2))
    img += blob * 255


for i in range(N_POS):
    img = np.zeros((SIZE, SIZE), np.float32)
    n_stars = random.randint(15, 35)
    records = []
    for _ in range(n_stars):
        cx = random.randint(10, SIZE-10)
        cy = random.randint(10, SIZE-10)
        sigma = random.uniform(1.0, 2.5)
        add_star(img, cx, cy, sigma)
        records.append([1, cx/SIZE, cy/SIZE, 0, 0, sigma])


    img += np.random.poisson(lam=2, size=img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)

    name = f"image_{i:03d}.png"
    cv2.imwrite(f"{OUT}/stars/{name}", img)
    pd.DataFrame(records).to_csv(f"{OUT}/stars/coords_{i:03d}.csv",
                                 header=False, index=False)


for i in range(N_NEG):
    img = np.zeros((SIZE, SIZE), np.float32)

    for _ in range(random.randint(3, 7)):
        cx = random.randint(20, SIZE-20)
        cy = random.randint(20, SIZE-20)
        rx = random.uniform(3, 12)
        ry = random.uniform(3, 12)
        add_blob(img, cx, cy, rx, ry)

    for _ in range(random.randint(2, 5)):
        x1, y1 = random.randint(0, SIZE), random.randint(0, SIZE)
        x2, y2 = x1+random.randint(-30,30), y1+random.randint(-30,30)
        cv2.line(img, (x1,y1), (x2,y2), 255, 1)

    img += np.random.poisson(lam=3, size=img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(f"{OUT}/neg/image_{i:03d}.png", img)

print("✔ Dataset hazır:")
print(f"  stars/: {N_POS} görüntü")
print(f"  neg/:   {N_NEG} görüntü")
