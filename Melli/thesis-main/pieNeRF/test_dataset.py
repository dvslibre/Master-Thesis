"""Utility script to load a SPECT YAML config, build the dataset, and visualize AP/PA/CT slices."""

# test_dataset.py
# -*- coding: utf-8 -*-
import yaml
import matplotlib.pyplot as plt
import numpy as np

from graf.config import get_data  # das überarbeitete config.py wird hier importiert


def main(config_path="configs/spect.yaml"):
    # 1️⃣ Config laden
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"\n✅ YAML geladen aus {config_path}")

    # 2️⃣ Dataset laden
    dataset, hwfr, render_poses = get_data(config)
    print(f"✅ Dataset geladen: {len(dataset)} Fälle")
    print(f"   -> hwfr: {hwfr}")
    print(f"   -> render_poses: {render_poses}")

    # 3️⃣ Beispiel-Fall prüfen
    sample = dataset[0]
    ap, pa, ct = sample["ap"], sample["pa"], sample["ct"]

    print("\n--- Beispiel-Fall ---")
    print(f"AP-Shape: {tuple(ap.shape)}, min={ap.min():.3f}, max={ap.max():.3f}")
    print(f"PA-Shape: {tuple(pa.shape)}, min={pa.min():.3f}, max={pa.max():.3f}")
    print(f"CT-Shape: {tuple(ct.shape)}, min={ct.min():.3f}, max={ct.max():.3f}")
    print(f"Meta: {sample['meta']}")

    # 4️⃣ Zu NumPy konvertieren
    ap_raw = ap.squeeze().numpy()      # [H,W]
    pa_raw = pa.squeeze().numpy()
    ct_vol = ct.numpy()                # [D,H,W] = [z,y,x]

    # 🔄 AP & PA: erst vertikal flippen (oben/unten tauschen), dann 90° CW
    ap_img = np.rot90(np.flipud(ap_raw), k=-1)
    pa_img = np.rot90(np.flipud(pa_raw), k=-1)

    # 🔄 CT: coronal Slice → 90° CW → vertikal + horizontal flippen
    mid_y = ct_vol.shape[1] // 2
    ct_coronal = ct_vol[:, mid_y, :]       # [D,W]
    ct_img = np.rot90(ct_coronal, k=-1)    # 90° CW
    ct_img = np.flipud(ct_img)             # vertikal
    ct_img = np.fliplr(ct_img)             # horizontal (Längsachse)

    # 5️⃣ Visualisierung
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(ap_img, cmap="gray")
    axs[0].set_title("AP projection")
    axs[0].axis("off")

    axs[1].imshow(pa_img, cmap="gray")
    axs[1].set_title("PA projection")
    axs[1].axis("off")

    axs[2].imshow(ct_img, cmap="gray")
    axs[2].set_title("CT (coronal)")
    axs[2].axis("off")

    plt.tight_layout()

    out_path = "data_check.png"
    plt.savefig(out_path, dpi=150)
    print(f"\n✅ Visualisierung gespeichert unter: {out_path}\n")


if __name__ == "__main__":
    main()
