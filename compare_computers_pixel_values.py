"""
Compare raw pixel values of the same RCF films scanned/processed on two
different computers.

CLEAR_dosimetry/filmcompare_newcomputer/ and CLEAR_dosimetry/film_compare/
each hold scans of the same physical films (same filenames in both). This
loads each film with the same crop/channel/marking-removal used in
CLEAR_RCF_processing.ipynb and compares pixel values directly -- no dose
conversion, no slice fitting.
"""

import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from pathlib import Path


def remove_marking(film):
    h, w = film.shape
    mask = np.ones((h, w), dtype=bool)
    mask[50:-50, 50:-50] = False
    film[(mask) & (film < 30000)] = 40000
    return film


def load_film(tif_path, channel=0, crop=(25, -25, 25, -25), clean_marking=True):
    film = tiff.imread(tif_path)[crop[0]:crop[1], crop[2]:crop[3], channel]
    if clean_marking:
        film = remove_marking(film)
    return film


def center_crop_to_match(a, b):
    """The two computers' scans aren't quite the same pixel dimensions;
    centre-crop both to their common overlap size before comparing."""
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])

    a0 = (a.shape[0] - h) // 2
    a1 = (a.shape[1] - w) // 2
    b0 = (b.shape[0] - h) // 2
    b1 = (b.shape[1] - w) // 2

    return a[a0:a0 + h, a1:a1 + w], b[b0:b0 + h, b1:b1 + w]


def align_images(a, b, max_shift=8):
    """Centre-crop to a common size, then find the small integer (dy, dx)
    shift of b that best aligns it to a by brute-force SSD search over
    +/-max_shift pixels (no known offset, but it's expected to be small --
    no skimage available in this environment for subpixel registration).
    Returns the aligned, equally-shaped crops of a and b, and the shift found.
    """
    a, b = center_crop_to_match(a, b)
    h, w = a.shape
    margin = max_shift

    a_t = a[margin:h - margin, margin:w - margin].astype(float)

    best_shift = (0, 0)
    best_score = np.inf
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            b_win = b[margin + dy:h - margin + dy, margin + dx:w - margin + dx].astype(float)
            score = np.sum((a_t - b_win) ** 2)
            if score < best_score:
                best_score = score
                best_shift = (dy, dx)

    dy, dx = best_shift
    a_aligned = a[margin:h - margin, margin:w - margin]
    b_aligned = b[margin + dy:h - margin + dy, margin + dx:w - margin + dx]
    return a_aligned, b_aligned, best_shift


def diff_stats(diff):
    """Statistics of the pixel-wise difference array (not a single collapsed
    mean/std -- the full spread, since a small mean can hide a wide or
    skewed per-pixel distribution)."""
    return {
        "mean": diff.mean(),
        "median": np.median(diff),
        "std": diff.std(),
        "mad": np.median(np.abs(diff - np.median(diff))),
        "p5": np.percentile(diff, 5),
        "p95": np.percentile(diff, 95),
        "min": diff.min(),
        "max": diff.max(),
    }


def add_hover_readout(fig, image_axes, film_a, film_b, diff, label_a, label_b):
    """Show the pixel under the cursor (row/col + value in each array)
    whenever the mouse is over any of the image panels."""
    readout = fig.text(
        0.5, 0.97, "", ha="center", va="top", fontsize=10, family="monospace",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"),
    )
    h, w = film_a.shape

    def on_move(event):
        if event.inaxes not in image_axes or event.xdata is None:
            readout.set_text("")
            fig.canvas.draw_idle()
            return
        col, row = int(round(event.xdata)), int(round(event.ydata))
        if 0 <= row < h and 0 <= col < w:
            readout.set_text(
                f"row={row}, col={col}  |  {label_a}={film_a[row, col]:.0f}  "
                f"{label_b}={film_b[row, col]:.0f}  diff={diff[row, col]:+.0f}"
            )
        else:
            readout.set_text("")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)


def compare_film(name, folder_a, folder_b, channel=0, label_a="new computer", label_b="old computer", max_shift=8):
    film_a = load_film(folder_a / f"{name}.tif", channel)
    film_b = load_film(folder_b / f"{name}.tif", channel)
    film_a, film_b, shift = align_images(film_a, film_b, max_shift=max_shift)

    diff = film_a.astype(float) - film_b.astype(float)
    vmax = np.abs(diff).max()
    stats = diff_stats(diff)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"{name}: pixel value comparison ({label_a} vs {label_b}), aligned shift (dy,dx)={shift}")

    im0 = axes[0, 0].imshow(film_a, cmap="gray", origin="lower")
    axes[0, 0].set_title(label_a)
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(film_b, cmap="gray", origin="lower")
    axes[0, 1].set_title(label_b)
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[0, 2].imshow(diff, cmap="RdBu_r", origin="lower", vmin=-vmax, vmax=vmax)
    axes[0, 2].set_title(f"{label_a} - {label_b}\nmedian={stats['median']:.1f}, MAD={stats['mad']:.1f}")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    sample = np.random.choice(film_a.size, size=min(20000, film_a.size), replace=False)
    axes[1, 0].scatter(film_a.ravel()[sample], film_b.ravel()[sample], s=2, alpha=0.3)
    lims = [min(film_a.min(), film_b.min()), max(film_a.max(), film_b.max())]
    axes[1, 0].plot(lims, lims, "k--", linewidth=1, label="y = x")
    axes[1, 0].set_xlabel(label_a)
    axes[1, 0].set_ylabel(label_b)
    axes[1, 0].legend()

    axes[1, 1].hist(diff.ravel(), bins=200)
    axes[1, 1].axvline(stats["median"], color="k", linestyle="--", linewidth=1, label="median")
    axes[1, 1].axvline(stats["p5"], color="gray", linestyle=":", linewidth=1, label="5th/95th pct")
    axes[1, 1].axvline(stats["p95"], color="gray", linestyle=":", linewidth=1)
    axes[1, 1].set_xlabel(f"{label_a} - {label_b} (per pixel)")
    axes[1, 1].set_ylabel("Pixel count")
    axes[1, 1].legend(fontsize=8)

    axes[1, 2].axis("off")
    stats_text = "\n".join(f"{k:>6s} = {v:+.1f}" for k, v in stats.items())
    axes[1, 2].text(0.05, 0.95, "Pixel-wise diff stats:\n\n" + stats_text,
                     va="top", ha="left", family="monospace", fontsize=11, transform=axes[1, 2].transAxes)

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    add_hover_readout(fig, [axes[0, 0], axes[0, 1], axes[0, 2]], film_a, film_b, diff, label_a, label_b)
    return fig, film_a, film_b, diff, shift, stats


if __name__ == "__main__":
    channel = 0
    folder_a = Path("CLEAR_dosimetry/filmcompare_newcomputer")
    folder_b = Path("CLEAR_dosimetry/film_compare")
    film_names = ["D_021", "D_022", "D_026", "D_027"]

    for name in film_names:
        fig, film_a, film_b, diff, shift, stats = compare_film(name, folder_a, folder_b, channel)
        print(f"{name}: shift(dy,dx)={shift}")
        for k, v in stats.items():
            print(f"    {k:>6s} = {v:+.2f}")
        plt.show()
