import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image


class FolderImageDataset(Dataset):
    def __init__(self, root_dir: str, tfm):
        self.root_dir = root_dir
        self.tfm = tfm
        self.exts = (".bmp", ".jpg", ".jpeg", ".png", ".webp")

        print(f"[INFO] Escaneando directorio: {root_dir}")

        self.class_names = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        if not self.class_names:
            raise RuntimeError(f"No se encontraron clases en: {root_dir}")

        print(f"[INFO] Clases encontradas ({len(self.class_names)}): {self.class_names}")

        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.samples = []

        for c in self.class_names:
            cdir = os.path.join(root_dir, c)
            files = [f for f in os.listdir(cdir) if f.lower().endswith(self.exts)]
            print(f"[INFO] {c}: {len(files)} imágenes")
            for f in sorted(files):
                self.samples.append((os.path.join(cdir, f), self.class_to_idx[c]))

        print(f"[INFO] Total imágenes cargadas: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        x = self.tfm(img)
        fname = os.path.basename(path)
        return x, fname, label


def build_resnet50(device: str):
    print("[INFO] Cargando ResNet50 preentrenada")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.eval()
    model.to(device)
    return model


def generar_embeddings_rps(
    carpeta_imgs: str = "datos_procesados/piedra_papel_tijera",
    salida_dir: str = "embeddings/RPS",
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
):
    os.makedirs(salida_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Input: {carpeta_imgs}")
    print(f"[INFO] Output: {salida_dir}")

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = FolderImageDataset(carpeta_imgs, tfm)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = build_resnet50(device)

    embeddings = []
    filenames = []
    labels = []

    print("[INFO] Extrayendo embeddings")
    with torch.no_grad():
        for i, (xb, names, y) in enumerate(dataloader, start=1):
            xb = xb.to(device)
            emb = model(xb).cpu().numpy()
            embeddings.append(emb)
            filenames.extend(list(names))
            labels.extend(y.numpy().tolist())
            print(f"[INFO] *** Cargando y generando embeddings ****")

    X = np.vstack(embeddings)
    y_true = np.array(labels, dtype=np.int64)

    print(f"[INFO] Embeddings shape: {X.shape}")

    np.save(os.path.join(salida_dir, "X_resnet50.npy"), X)
    np.save(os.path.join(salida_dir, "y_true.npy"), y_true)

    with open(os.path.join(salida_dir, "filenames.txt"), "w") as f:
        for n in filenames:
            f.write(n + "\n")

    with open(os.path.join(salida_dir, "classes.txt"), "w") as f:
        for c in dataset.class_names:
            f.write(c + "\n")

    print("[INFO] Embeddings RPS guardados correctamente")

    return {
        "X_shape": X.shape,
        "num_images": len(dataset),
        "classes": dataset.class_names,
        "output_dir": salida_dir,
    }
