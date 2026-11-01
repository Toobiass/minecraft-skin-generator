"""Web demo: tags in, skins out, with a 3D preview. Runs locally in the browser.

    pip install -r demo/requirements.txt
    python export_weights.py         # once after training -> weights/
    python demo/app.py               # http://127.0.0.1:7860

MODEL=path/to/weights picks another weights folder.
"""
import base64
import html
import io
import os
import sys
import tempfile

import gradio as gr
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(HERE))   # skingen/ is one folder up

from skingen import config as C
from skingen.data import prompt_to_vec
from skingen.weights import load_weights
from skingen.maskgit import generate
from skingen.utils import to_skin

MODEL = os.environ.get("MODEL") or os.path.join(os.path.dirname(HERE), "weights")
MAX_SKINS = 4   # on a laptop CPU one skin takes ~10-20 s
EXAMPLES = [
    ["pirate"],
    ["girl", "pirate", "cute", "slim"],
    ["boy", "blue hoodie", "blond hair", "blue eyes", "classic"],
    ["anime", "boy", "white hair", "winter"],
    ["creeper"],
]

# 3D preview: skinview3d (three.js) in an iframe - gr.HTML does not run <script> tags,
# an iframe's srcdoc does.
VIEWER = """<!DOCTYPE html><html><body style="margin:0;background:transparent">
<canvas id="c"></canvas>
<script src="https://cdn.jsdelivr.net/npm/skinview3d@3/bundles/skinview3d.bundle.js"></script>
<script>
const v = new skinview3d.SkinViewer({canvas: document.getElementById("c"), width: 260,
                                     height: 340, skin: "%s", model: "%s"});
v.animation = new skinview3d.WalkingAnimation();
v.animation.speed = 0.6;
v.autoRotate = true;
v.zoom = 0.8;
</script></body></html>"""


def viewer_html(skins, arm_model):
    frames = []
    for s in skins:
        buf = io.BytesIO()
        s.save(buf, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        page = html.escape(VIEWER % (url, arm_model), quote=True)
        frames.append(f'<iframe srcdoc="{page}" width="260" height="340" '
                      f'style="border:0" title="3D skin"></iframe>')
    return '<div style="display:flex;flex-wrap:wrap;gap:8px;justify-content:center">'            + "".join(frames) + "</div>"


model, vq, vocab = load_weights(MODEL)
EXAMPLES = [[e] for e in EXAMPLES if all(t in vocab for t in e)]   # the dropdown rejects unknown tags


def run(tags, n, guidance, steps, seed):
    seed = int(seed) if seed is not None and seed >= 0 else int(torch.randint(0, 2**31 - 1, (1,)))
    torch.manual_seed(seed)
    y, _ = prompt_to_vec(",".join(tags or []), vocab)   # the dropdown only offers known tags
    toks = generate(model, y.to(C.DEVICE).repeat(int(n), 1), steps=int(steps), guidance=guidance)
    skins = [to_skin(im) for im in vq.decode(toks)]

    out = tempfile.mkdtemp()
    files = []
    for i, s in enumerate(skins):
        files.append(os.path.join(out, f"skin_{seed}_{i}.png"))
        s.save(files[-1])
    previews = [s.resize((512, 512), Image.NEAREST) for s in skins]   # pixel art: no smoothing
    tags = tags or []
    arm = "slim" if "slim" in tags else "default" if "classic" in tags else "auto-detect"
    return viewer_html(skins, arm), previews, files, seed


with gr.Blocks(title="Minecraft Skin Generator") as demo:
    gr.Markdown("# minecraft-skin-generator\nMinecraft skins from tags. VQ-VAE + MaskGIT, trained from "
                "scratch, no pretrained model. `classic` / `slim` picks the arm model.")
    with gr.Row():
        with gr.Column():
            tags = gr.Dropdown(vocab, multiselect=True, label="Tags",
                               info="Empty = no condition. Bare colours don't exist: use 'red hair', 'blue hoodie'.")
            n = gr.Slider(1, MAX_SKINS, value=2, step=1, label="Number of skins")
            with gr.Accordion("Settings", open=False):
                guidance = gr.Slider(1.0, 4.0, value=C.GUIDANCE, step=0.1, label="Guidance",
                                     info="How strongly the tags count. 1.5-2.0 works best.")
                steps = gr.Slider(8, 48, value=C.GEN_STEPS, step=1, label="Steps")
                seed = gr.Number(-1, precision=0, label="Seed (-1 = random)")
            btn = gr.Button("Generate", variant="primary")
            gr.Examples(EXAMPLES, inputs=[tags])
        with gr.Column():
            view3d = gr.HTML('<div style="height:340px;display:flex;align-items:center;'
                             'justify-content:center;opacity:.5">The 3D preview appears here</div>',
                             min_height=340)
            gallery = gr.Gallery(label="Textures", columns=2, format="png")
            files = gr.File(label="Skins (64x64 PNG, load them in the game)", file_count="multiple")
            used_seed = gr.Number(label="Seed used", precision=0, interactive=False)
    btn.click(run, [tags, n, guidance, steps, seed], [view3d, gallery, files, used_seed],
              show_progress_on=view3d)   # one spinner instead of one per output

if __name__ == "__main__":
    demo.launch()
