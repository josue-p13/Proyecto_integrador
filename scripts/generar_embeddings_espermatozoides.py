import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image


class FolderImageDataset(Dataset):
    """
    Espera estructura:
      root_dir/
        clase1/*.bmp|png|jpg
        clase2/*.bmp|png|jpg

    Devuelve:
      (tensor, filename, label_idx)
    """
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
            raise RuntimeError(f"No se encontraron subcarpetas de clase en: {root_dir}")

        print(f"[INFO] Clases encontradas ({len(self.class_names)}): {self.class_names}")

        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.samples = []

        for c in self.class_names:
            cdir = os.path.join(root_dir, c)
            archivos = [
                f for f in os.listdir(cdir)
                if f.lower().endswith(self.exts)
            ]
            print(f"[INFO] {c}: {len(archivos)} imágenes")
            for f in sorted(archivos):
                self.samples.append((os.path.join(cdir, f), self.class_to_idx[c]))

        if not self.samples:
            raise RuntimeError(f"No se encontraron imágenes válidas en: {root_dir}")

        print(f"[INFO] Total de imágenes cargadas: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        x = self.tfm(img)
        fname = os.path.basename(path)
        return x, fname, label


def build_resnet50_extractor(device: str) -> nn.Module:
    print("[INFO] Cargando ResNet50 preentrenada (ImageNet)")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Identity()  # embedding de 2048 dimensiones
    model.eval()
    model.to(device)
    print("[INFO] Modelo listo en modo evaluación")
    return model


def generar_embeddings_espermatozoides(
    carpeta_imgs: str = "datos_procesados/espermatozoides",
    salida_dir: str = "embeddings/Espermatozoides",
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
):
    os.makedirs(salida_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Dispositivo seleccionado: {device}")
    print(f"[INFO] Carpeta de entrada: {carpeta_imgs}")
    print(f"[INFO] Carpeta de salida: {salida_dir}")

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = FolderImageDataset(carpeta_imgs, tfm)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = build_resnet50_extractor(device)

    all_embeddings = []
    all_filenames = []
    all_labels = []

    print("[INFO] Iniciando extracción de embeddings")
    with torch.no_grad():
        for i, (xb, names, labels) in enumerate(dataloader, start=1):
            xb = xb.to(device)
            emb = model(xb).cpu().numpy()
            all_embeddings.append(emb)
            all_filenames.extend(list(names))
            all_labels.extend(labels.numpy().tolist())

            print(f"[INFO] *** Cargando y generando embeddings ****")

    X = np.vstack(all_embeddings)
    y_true = np.array(all_labels, dtype=np.int64)

    print(f"[INFO] Embeddings generados con forma: {X.shape}")

    np.save(os.path.join(salida_dir, "X_resnet50.npy"), X)
    np.save(os.path.join(salida_dir, "y_true.npy"), y_true)

    with open(os.path.join(salida_dir, "filenames.txt"), "w", encoding="utf-8") as f:
        for n in all_filenames:
            f.write(n + "\n")

    with open(os.path.join(salida_dir, "classes.txt"), "w", encoding="utf-8") as f:
        for c in dataset.class_names:
            f.write(c + "\n")

    print("[INFO] Archivos guardados:")
    print("       - X_resnet50.npy")
    print("       - y_true.npy")
    print("       - filenames.txt")
    print("       - classes.txt")

    print("[INFO] Extracción de embeddings finalizada correctamente")

    return {
        "X_shape": X.shape,
        "num_images": len(dataset),
        "num_classes": len(dataset.class_names),
        "output_dir": salida_dir,
    }
