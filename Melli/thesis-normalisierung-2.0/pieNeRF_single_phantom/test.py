import numpy as np
import matplotlib.pyplot as plt

act = np.load("data/phantom_01/out/act.npy")
assert act.ndim == 3, f"act.npy muss 3D sein, ist aber {act.ndim}D mit shape={act.shape}"

A, B, C = act.shape  # axis0, axis1, axis2
idx_list = [80, 128, 200]

out_path = "gt_sagittal_axis0_slices.png"

# Bounds check
for idx in idx_list:
    if idx < 0 or idx >= A:
        raise ValueError(f"axis0 idx={idx} out of bounds fuer A={A}")

# gemeinsame Ticks (gleiche Schrittweite, nicht zwingend gleiche Anzahl)
y_step = max(1, B // 10)
z_step = max(1, C // 10)
y_ticks = np.arange(0, B, y_step)
z_ticks = np.arange(0, C, z_step)

fig, axs = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)

last_im = None
for ax, idx in zip(axs, idx_list):
    # sagittal bei fix axis0 -> (axis1, axis2)
    img = act[idx, :, :].astype(np.float32)

    # optional: falls du eine "Kopf oben" Orientierung willst, kannst du hier flip/rot einbauen.
    # Beispiel:
    # img = np.flipud(img)  # falls y-Achse auf dem Kopf steht
    # img = np.fliplr(img)  # falls links/rechts gespiegelt ist

    # Wichtig: aspect=1 => gleiche Schrittweite / gleiche Pixelproportionen
    last_im = ax.imshow(
        img,
        origin="upper",
        extent=[0, C - 1, B - 1, 0],  # x-axis: z (axis2), y-axis: y (axis1)
        aspect="equal",
    )

    ax.set_title(f"GT sagittal (act) @ axis0={idx}")
    ax.set_xlabel("axis2 (z-like)")
    ax.set_ylabel("axis1 (y-like)")
    ax.set_xticks(z_ticks)
    ax.set_yticks(y_ticks)

# gemeinsame Colorbar
cbar = fig.colorbar(last_im, ax=axs, shrink=0.9)
cbar.set_label("Activity (raw units)")

plt.savefig(out_path, dpi=200)
plt.close()
print(f"[OUT] Saved {out_path}  (act.shape={act.shape})")