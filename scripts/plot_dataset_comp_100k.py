from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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


def _fmt_int(n: int) -> str:
    return f"{int(n):,}"


def _panel(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    *,
    title: str,
    title_font: ImageFont.ImageFont,
    border: bool = True,
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    if border:
        draw.rounded_rectangle([x0, y0, x1, y1], radius=18, outline=(200, 200, 200), width=2)
    # title
    tx = x0 + 18
    ty = y0 + 12
    draw.text((tx, ty), title, font=title_font, fill=(20, 20, 20))
    # content box (leave space for title)
    title_h = title_font.getbbox(title)[3] - title_font.getbbox(title)[1]
    return (x0 + 18, y0 + 12 + title_h + 12, x1 - 18, y1 - 18)


def _draw_bar_chart(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    *,
    labels: Sequence[str],
    values: Sequence[int],
    total: int | None = None,
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
    bar_color: Tuple[int, int, int] = (74, 144, 226),
    show_percent: bool = True,
) -> None:
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    if w <= 0 or h <= 0:
        return

    # layout
    left_pad = 10
    right_pad = 10
    top_pad = 10
    bottom_pad = 46  # for x labels
    px0 = x0 + left_pad
    px1 = x1 - right_pad
    py0 = y0 + top_pad
    py1 = y1 - bottom_pad
    if px1 <= px0 or py1 <= py0:
        return

    max_v = max([int(v) for v in values] + [1])
    n = max(1, len(labels))
    gap = max(8, int((px1 - px0) * 0.02))
    bar_w = max(6, int((px1 - px0 - gap * (n + 1)) / n))

    # axes
    axis_col = (120, 120, 120)
    draw.line([(px0, py1), (px1, py1)], fill=axis_col, width=2)
    draw.line([(px0, py0), (px0, py1)], fill=axis_col, width=2)

    for i, (lab, v) in enumerate(zip(labels, values)):
        v = int(v)
        bx0 = px0 + gap + i * (bar_w + gap)
        bx1 = bx0 + bar_w
        bar_h = int((v / max_v) * (py1 - py0))
        by0 = py1 - bar_h
        draw.rounded_rectangle([bx0, by0, bx1, py1], radius=8, fill=bar_color)

        # value label
        if total is not None and show_percent and total > 0:
            s = f"{_fmt_int(v)} ({v / total * 100:.1f}%)"
        else:
            s = _fmt_int(v)
        tw = small_font.getbbox(s)[2] - small_font.getbbox(s)[0]
        draw.text((bx0 + (bar_w - tw) // 2, max(py0, by0 - 20)), s, font=small_font, fill=(30, 30, 30))

        # x label
        lab_s = str(lab)
        tw2 = font.getbbox(lab_s)[2] - font.getbbox(lab_s)[0]
        draw.text((bx0 + (bar_w - tw2) // 2, py1 + 10), lab_s, font=font, fill=(30, 30, 30))


def _parse_k_top(k_items: Sequence[Sequence[object]], *, max_items: int = 10) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    for item in list(k_items)[: int(max_items)]:
        if not item or len(item) < 2:
            continue
        k_tok = str(item[0])
        cnt = int(item[1])
        # shorten "<K_6>" -> "K6"
        k_short = k_tok.strip("<>").replace("_", "")
        out.append((k_short, cnt))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot dataset_comp_100k_summary.json stats to a single PNG (no matplotlib required).")
    p.add_argument("--summary", type=Path, default=Path("outputs/dataset_comp_100k_summary.json"), help="Summary JSON file.")
    p.add_argument("--out", type=Path, default=Path("outputs/dataset_comp_100k_summary.png"), help="Output PNG path.")
    p.add_argument("--width", type=int, default=2200, help="Image width in pixels.")
    p.add_argument("--height", type=int, default=1500, help="Image height in pixels.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text())
    train = summary["train"]
    val = summary["val"]

    train_total = int(train["num_samples"])
    val_total = int(val["num_samples"])

    train_ft = dict(train["filter_type_counts"])
    val_ft = dict(val["filter_type_counts"])

    # The summary JSON in this repo does not currently include scenario/order counts, so
    # we use the numbers from the 100K dataset analysis note (can be overridden later).
    train_scenario = {
        "general": 34568,
        "anti_jamming": 20134,
        "coexistence": 20050,
        "wideband_rejection": 15013,
        "random_basic": 10235,
    }
    order6 = 20437
    order7 = 20401
    train_order = {
        "order=6": order6,
        "order=7": order7,
        "others": max(0, train_total - order6 - order7),
    }

    # Prepare K distributions per filter type (top list from summary).
    k_by_type: Dict[str, List[Tuple[str, int]]] = {}
    for ftype in ("bandpass", "bandstop", "highpass", "lowpass"):
        top = train["by_filter_type"][ftype]["k_dist_top"]
        items = _parse_k_top(top, max_items=12)
        known_sum = sum(c for _, c in items)
        other = max(0, int(train["by_filter_type"][ftype]["count"]) - known_sum)
        if other > 0:
            items.append(("other", other))
        k_by_type[ftype] = items

    # ---- draw ----
    img = Image.new("RGB", (int(args.width), int(args.height)), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    title_font = _load_font(36)
    panel_title_font = _load_font(28)
    font = _load_font(18)
    small_font = _load_font(16)

    margin = 60
    gap = 60
    W, H = img.size
    col_w = (W - 2 * margin - gap) // 2
    row1_h = 350
    row2_h = 350
    row3_h = H - 2 * margin - 2 * gap - row1_h - row2_h

    # global title
    main_title = "100K 数据集统计概览（train/val）"
    tw = title_font.getbbox(main_title)[2] - title_font.getbbox(main_title)[0]
    draw.text(((W - tw) // 2, 10), main_title, font=title_font, fill=(0, 0, 0))

    y = margin
    # row 1
    b1 = (margin, y, margin + col_w, y + row1_h)
    b2 = (margin + col_w + gap, y, margin + col_w + gap + col_w, y + row1_h)
    c1 = _panel(draw, b1, title=f"Train filter_type（N={_fmt_int(train_total)}）", title_font=panel_title_font)
    c2 = _panel(draw, b2, title=f"Val filter_type（N={_fmt_int(val_total)}）", title_font=panel_title_font)
    _draw_bar_chart(
        draw,
        c1,
        labels=list(train_ft.keys()),
        values=[int(train_ft[k]) for k in train_ft.keys()],
        total=train_total,
        font=font,
        small_font=small_font,
        bar_color=(64, 128, 255),
    )
    _draw_bar_chart(
        draw,
        c2,
        labels=list(val_ft.keys()),
        values=[int(val_ft[k]) for k in val_ft.keys()],
        total=val_total,
        font=font,
        small_font=small_font,
        bar_color=(90, 180, 90),
    )

    # row 2
    y += row1_h + gap
    b3 = (margin, y, margin + col_w, y + row2_h)
    b4 = (margin + col_w + gap, y, margin + col_w + gap + col_w, y + row2_h)
    c3 = _panel(draw, b3, title="Train scenario 分布", title_font=panel_title_font)
    c4 = _panel(draw, b4, title="Train order（仅展示 6/7 + 其它）", title_font=panel_title_font)
    _draw_bar_chart(
        draw,
        c3,
        labels=list(train_scenario.keys()),
        values=[int(train_scenario[k]) for k in train_scenario.keys()],
        total=train_total,
        font=font,
        small_font=small_font,
        bar_color=(255, 170, 64),
    )
    _draw_bar_chart(
        draw,
        c4,
        labels=list(train_order.keys()),
        values=[int(train_order[k]) for k in train_order.keys()],
        total=train_total,
        font=font,
        small_font=small_font,
        bar_color=(180, 120, 240),
    )

    # row 3: K distribution (small multiples)
    y += row2_h + gap
    b5 = (margin, y, W - margin, y + row3_h)
    c5 = _panel(draw, b5, title="DSL repeat 阶数 <K_*> 分布（按 filter_type, Top + other）", title_font=panel_title_font)
    cx0, cy0, cx1, cy1 = c5
    inner_gap = 40
    sub_w = (cx1 - cx0 - inner_gap) // 2
    sub_h = (cy1 - cy0 - inner_gap) // 2

    ftypes = ["bandpass", "bandstop", "highpass", "lowpass"]
    for idx, ftype in enumerate(ftypes):
        r = idx // 2
        c = idx % 2
        sx0 = cx0 + c * (sub_w + inner_gap)
        sy0 = cy0 + r * (sub_h + inner_gap)
        sx1 = sx0 + sub_w
        sy1 = sy0 + sub_h
        # sub-panel border + title
        draw.rounded_rectangle([sx0, sy0, sx1, sy1], radius=14, outline=(220, 220, 220), width=2)
        sub_title = f"{ftype}（N={_fmt_int(int(train['by_filter_type'][ftype]['count']))}）"
        draw.text((sx0 + 12, sy0 + 10), sub_title, font=_load_font(20), fill=(20, 20, 20))
        title_h = _load_font(20).getbbox(sub_title)[3] - _load_font(20).getbbox(sub_title)[1]
        sub_box = (sx0 + 12, sy0 + 10 + title_h + 10, sx1 - 12, sy1 - 12)
        items = k_by_type[ftype]
        _draw_bar_chart(
            draw,
            sub_box,
            labels=[k for k, _ in items],
            values=[v for _, v in items],
            total=int(train["by_filter_type"][ftype]["count"]),
            font=_load_font(14),
            small_font=_load_font(12),
            bar_color=(80, 160, 160),
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()

