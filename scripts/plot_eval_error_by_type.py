from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quick bar chart: mean/median best_error by filter type (PIL).")
    p.add_argument("--out", type=Path, default=Path("outputs/eval_error_by_type.png"), help="Output PNG path.")
    p.add_argument("--width", type=int, default=1100)
    p.add_argument("--height", type=int, default=520)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    stats = [
        {"type": "bandpass", "mean": 0.0416, "median": 0.0264, "n": 83},
        {"type": "bandstop", "mean": 0.6194, "median": 0.7977, "n": 36},
        {"type": "highpass", "mean": 0.4600, "median": 0.4592, "n": 37},
        {"type": "lowpass", "mean": 0.5090, "median": 0.5060, "n": 24},
    ]

    W, H = int(args.width), int(args.height)
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    title_font = _load_font(32)
    font = _load_font(18)
    small = _load_font(15)

    title = "不同滤波器类型的 best_error（mean vs median）"
    tw = title_font.getbbox(title)[2]
    draw.text(((W - tw) // 2, 18), title, font=title_font, fill=(0, 0, 0))

    # plot area
    margin_l, margin_r = 90, 40
    margin_t, margin_b = 90, 90
    x0, y0 = margin_l, margin_t
    x1, y1 = W - margin_r, H - margin_b

    # axes
    axis = (120, 120, 120)
    draw.line([(x0, y1), (x1, y1)], fill=axis, width=2)
    draw.line([(x0, y0), (x0, y1)], fill=axis, width=2)

    # y scale
    max_v = max(max(s["mean"], s["median"]) for s in stats)
    y_max = max(0.1, max_v * 1.15)
    y_ticks = 5
    grid = (235, 235, 235)
    for i in range(y_ticks + 1):
        v = y_max * i / y_ticks
        yy = y1 - int((v / y_max) * (y1 - y0))
        draw.line([(x0, yy), (x1, yy)], fill=grid, width=1)
        label = f"{v:.2f}"
        bw = small.getbbox(label)[2]
        draw.text((x0 - 12 - bw, yy - 8), label, font=small, fill=(60, 60, 60))

    # bars
    n = len(stats)
    gap = 26
    group_w = (x1 - x0 - gap * (n + 1)) // n
    bar_gap = max(8, group_w // 8)
    bar_w = (group_w - bar_gap) // 2
    colors = {"mean": (64, 128, 255), "median": (255, 170, 64)}

    for i, s in enumerate(stats):
        gx0 = x0 + gap + i * (group_w + gap)
        # mean bar
        for j, key in enumerate(("mean", "median")):
            v = float(s[key])
            bx0 = gx0 + j * (bar_w + bar_gap)
            bx1 = bx0 + bar_w
            bh = int((v / y_max) * (y1 - y0))
            by0 = y1 - bh
            draw.rounded_rectangle([bx0, by0, bx1, y1], radius=8, fill=colors[key])
            val_s = f"{v:.3f}"
            twv = small.getbbox(val_s)[2]
            draw.text((bx0 + (bar_w - twv) // 2, max(y0, by0 - 20)), val_s, font=small, fill=(40, 40, 40))

        # x label
        lab = f"{s['type']}\n(n={s['n']})"
        # crude multi-line centering
        lines = lab.split("\n")
        bbox = font.getbbox("Ag")
        line_h = bbox[3] - bbox[1]
        _ = line_h * len(lines) + 2 * (len(lines) - 1)
        y_text = y1 + 14
        for li, line in enumerate(lines):
            tww = font.getbbox(line)[2]
            draw.text((gx0 + (group_w - tww) // 2, y_text + li * (line_h + 2)), line, font=font, fill=(30, 30, 30))

    # legend
    lx, ly = x1 - 260, y0 - 52
    legend_items = [("mean", "mean"), ("median", "median")]
    for i, (key, label) in enumerate(legend_items):
        cx = lx + i * 120
        draw.rounded_rectangle([cx, ly, cx + 22, ly + 22], radius=5, fill=colors[key])
        draw.text((cx + 30, ly + 1), label, font=font, fill=(30, 30, 30))

    # axis labels
    draw.text((x0, y0 - 44), "best_error", font=font, fill=(60, 60, 60))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
