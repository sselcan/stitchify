"""Web demo for stitchify: upload a photo, get a cross stitch chart."""
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

import stitchify as st


def make_pattern(photo, width, colors, bg_strength, focus_subject):
    if photo is None:
        raise gr.Error("Please upload a photo first.")
    img = np.array(photo.convert("RGB"))
    scale = 800 / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    if focus_subject:
        img = st.simplify_background(img, st.subject_mask(img), bg_strength)

    h = round(img.shape[0] * width / img.shape[1])
    grid = cv2.resize(img, (int(width), h), interpolation=cv2.INTER_AREA).astype(np.float64)
    idx, cols = st.quantize_to_dmc(grid, int(colors))
    idx = st.remove_confetti(idx)

    prefix = str(Path(tempfile.mkdtemp()) / "pattern")
    st.render(idx, cols, prefix)

    legend = Path(f"{prefix}_legend.tsv").read_text().splitlines()
    table = [row.split("\t") for row in legend[1:]]
    return f"{prefix}_preview.png", f"{prefix}_chart.png", table, st.metrics(idx)


with gr.Blocks(title="stitchify") as demo:
    gr.Markdown("# stitchify\nTurn a photo into a simplified cross stitch pattern. "
                "Photos are processed temporarily and not stored.")
    with gr.Row():
        with gr.Column():
            photo = gr.Image(type="pil", label="Your photo")
            width = gr.Slider(30, 150, value=80, step=5, label="Width (stitches)")
            colors = gr.Slider(3, 30, value=12, step=1, label="Max thread colors")
            focus = gr.Checkbox(value=True, label="Focus on main subject (simplify background)")
            bg = gr.Slider(0.5, 3.0, value=1.5, step=0.25, label="Background simplification")
            go = gr.Button("Make pattern", variant="primary")
        with gr.Column():
            preview = gr.Image(label="Preview")
            chart = gr.Image(label="Chart (download with the button in the corner)")
    legend = gr.Dataframe(headers=["symbol", "DMC", "name", "stitches"], label="Threads")
    stats = gr.JSON(label="Stitchability metrics")

    go.click(make_pattern, [photo, width, colors, bg, focus], [preview, chart, legend, stats])

if __name__ == "__main__":
    demo.launch()
