"""
Web-based point-cloud browser for a video-eval output tree.

    python -m src.scripts.pcd_web_viewer \
        --dataset re10k=outputs/video_eval_re10k_release_video_ft_step100k \
        --dataset dl3dv=outputs/video_eval_dl3dv_release_video_ft_step100k \
        --port <PORT>

Then open http://localhost:<PORT> (tunnel the port first if data are on a remote machine:
`ssh -N -L <PORT>:localhost:<PORT> <host>`). It reads each scene from disk on request and unprojects
it to a point cloud in the browser; the fitted (a, b) scale from scale_fit.json can be adjusted
live (see src/scripts/fit_context_scale.py for how that fit is produced).
"""

import argparse
import base64
import io
import json
import struct
import threading
import zlib
from collections import OrderedDict
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).parent
INDEX_HTML = HERE / "pcd_web_viewer.html"

CLOUD_CACHE_SIZE = 6   # built point clouds kept in memory, so revisiting a scene is instant
OPACITY_THRESHOLD = 0.5   # below this, the rasterized depth is not trustworthy


# --------------------------------------------------------------------------------------------
# Scene loading
# --------------------------------------------------------------------------------------------

def list_scenes(root: Path):
    """Scene directories that actually carry the data we need, sorted by id."""
    if not root.is_dir():
        return []
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if (d / "cam_dict/cameras.torch").exists() and (d / "depth").is_dir():
            out.append(d.name)
    return out


def load_scene_meta(scene_dir: Path):
    cam = torch.load(scene_dir / "cam_dict/cameras.torch", map_location="cpu")
    target_index = cam["target_index"].tolist()
    context_index = cam["context_index"].tolist()
    extrinsics = cam["extrinsics"].float()          # (V,4,4) camera-to-world
    intrinsics = cam["intrinsics"].float()          # (V,3,3) normalized

    origins = extrinsics[:, :3, 3]   # camera centers, one per frame

    # World-space "down"/"forward" (OpenCV convention: c2w column 1 is +y/down, column 2 is +z/into
    # the scene), used by the browser to auto-upright the cloud.
    down = extrinsics[:, :3, 1].mean(0)
    forward = extrinsics[:, :3, 2].mean(0)
    down = down / down.norm().clamp_min(1e-8)
    forward = forward / forward.norm().clamp_min(1e-8)

    fit_path = scene_dir / "scale_fit.json"
    fit = json.loads(fit_path.read_text()) if fit_path.exists() else None

    sample = torch.load(scene_dir / "depth" / f"{target_index[0]:0>6}.pt", map_location="cpu")
    h, w = sample.shape[-2:]

    return {
        "target_index": target_index,
        "context_index": context_index,
        "n_frames": len(target_index),
        "height": int(h),
        "width": int(w),
        "near": cam["near"].tolist(),
        "far": cam["far"].tolist(),
        "origins": origins.tolist(),
        "down": down.tolist(),
        "forward": forward.tolist(),
        "fit": fit,
        "has_rast": (scene_dir / "depth_rast/rasterized.torch").exists(),
        "extrinsics": extrinsics,   # tensors here for build_cloud; /api/scene converts to lists
        "intrinsics": intrinsics,
    }


def to_z01(depth: torch.Tensor, convention: str) -> torch.Tensor:
    """Normalize stored depth to [0,1], whichever convention the eval run used."""
    if convention == "pm1":
        return (depth.clamp(-1.0, 1.0) + 1.0) * 0.5
    return depth.clamp(0.0, 1.0)


def detect_convention(depth: torch.Tensor, meta) -> str:
    if meta.get("fit") and meta["fit"].get("convention"):
        return meta["fit"]["convention"]
    return "pm1" if float(depth.min()) < -0.05 else "01"


def build_cloud(scene_dir: Path, meta, stride: int):
    """Unproject every frame to world-space rays and pack them into one binary blob (bytes, info).
    Points are stored as (direction, z01) rather than positions, so the browser can recover a
    position for any (a, b) with a single multiply-add.
    """
    target_index = meta["target_index"]
    extrinsics, intrinsics = meta["extrinsics"], meta["intrinsics"]
    h, w = meta["height"], meta["width"]

    rast = None
    if meta["has_rast"]:
        blob = torch.load(scene_dir / "depth_rast/rasterized.torch", map_location="cpu")
        rast = blob["mask"].float()                 # (V,H,W) opacity

    dirs_all, z_all, rgb_all, frame_all, edge_all, rmask_all, sky_all = [], [], [], [], [], [], []
    convention = None
    fars = meta["far"]

    for fi, idx in enumerate(target_index):
        depth = torch.load(scene_dir / "depth" / f"{idx:0>6}.pt", map_location="cpu").float()
        if convention is None:
            convention = detect_convention(depth, meta)
        z01 = to_z01(depth, convention)

        # Depth-gradient magnitude (before subsampling): flying-pixel artifacts sit on depth edges,
        # so thresholding this in the browser removes them.
        gy = torch.zeros_like(z01)
        gx = torch.zeros_like(z01)
        gy[1:-1, :] = (z01[2:, :] - z01[:-2, :]).abs() * 0.5
        gx[:, 1:-1] = (z01[:, 2:] - z01[:, :-2]).abs() * 0.5
        edge = torch.sqrt(gx * gx + gy * gy)

        color_path = scene_dir / "color" / f"{idx:0>6}.png"
        if color_path.exists():
            img = Image.open(color_path).convert("RGB").resize((w, h), Image.BILINEAR)
            rgb = torch.from_numpy(np.asarray(img).copy())        # (H,W,3) uint8
        else:
            rgb = torch.full((h, w, 3), 128, dtype=torch.uint8)

        # Pixel grid -> camera-frame rays: x=(u-cx)*z/fx, y=(v-cy)*z/fy, z=z.
        K = intrinsics[fi]
        fx, fy = float(K[0, 0]) * w, float(K[1, 1]) * h
        cx, cy = float(K[0, 2]) * w, float(K[1, 2]) * h

        vs = torch.arange(0, h, stride, dtype=torch.float32)
        us = torch.arange(0, w, stride, dtype=torch.float32)
        gv, gu = torch.meshgrid(vs, us, indexing="ij")
        d_cam = torch.stack([(gu - cx) / fx, (gv - cy) / fy, torch.ones_like(gu)], dim=-1)

        R = extrinsics[fi, :3, :3]
        d_world = d_cam.reshape(-1, 3) @ R.T                      # (N,3) world-space ray directions

        sub = (slice(None, None, stride), slice(None, None, stride))
        dirs_all.append(d_world)
        z_all.append(z01[sub].reshape(-1))
        edge_all.append(edge[sub].reshape(-1))
        rgb_all.append(rgb[sub].reshape(-1, 3))
        frame_all.append(torch.full((d_world.shape[0],), fi, dtype=torch.int32))
        if rast is not None:
            rmask_all.append((rast[fi][sub].reshape(-1) > OPACITY_THRESHOLD).to(torch.uint8) * 255)
        else:
            rmask_all.append(torch.full((d_world.shape[0],), 255, dtype=torch.uint8))

        # Sky / unbounded-depth flag, from the pseudo-GT (the prediction itself is unreliable there).
        gt_path = scene_dir / "depth_gt" / f"{idx:0>6}.pt"
        if gt_path.exists():
            g = torch.load(gt_path, map_location="cpu").float()[sub].reshape(-1)
            sky = (~torch.isfinite(g)) | (g <= 0) | (g > float(fars[fi]))
            sky_all.append(sky.to(torch.uint8) * 255)
        else:
            sky_all.append(torch.zeros(d_world.shape[0], dtype=torch.uint8))

    dirs = torch.cat(dirs_all).numpy().astype("<f4")
    z = torch.cat(z_all).numpy().astype("<f4")
    edge = torch.cat(edge_all).numpy().astype("<f4")
    frame = torch.cat(frame_all).numpy().astype("<u2")
    rgb = torch.cat(rgb_all).numpy().astype("<u1")
    rmask = torch.cat(rmask_all).numpy().astype("<u1")
    sky = torch.cat(sky_all).numpy().astype("<u1")
    n = int(z.shape[0])

    # PCD2: float fields first so every 4-byte field stays aligned in the browser's ArrayBuffer.
    payload = b"".join([
        b"PCD2",
        struct.pack("<II", n, 1 if rast is not None else 0),
        dirs.tobytes(), z.tobytes(), edge.tobytes(),
        frame.tobytes(), rgb.tobytes(), rmask.tobytes(), sky.tobytes(),
    ])
    return payload, {"n_points": n, "convention": convention,
                     "sky_points": int((sky > 127).sum()),
                     "sky_fraction": float((sky > 127).mean())}


# --------------------------------------------------------------------------------------------
# Server
# --------------------------------------------------------------------------------------------

# Composite recording layout: white ground, cloud in the middle, inputs on the left, predictions
# on the right.
COMPOSE_BG = (255, 255, 255)
COMPOSE_LABEL = (28, 30, 34)
COMPOSE_BORDER = (203, 207, 213)


def _label_font(px: int):
    """DejaVuSans-Bold, which ships with matplotlib (already a project dependency)."""
    from PIL import ImageFont
    try:
        import matplotlib
        return ImageFont.truetype(
            str(Path(matplotlib.__file__).parent / "mpl-data/fonts/ttf/DejaVuSans-Bold.ttf"), px)
    except Exception:
        return ImageFont.load_default()


def _stack_cell(img: Image.Image, count: int, avail_h: int, max_w: int, gap: int):
    """Cell size for `count` copies of `img`'s aspect stacked vertically inside avail_h."""
    ch = (avail_h - gap * (count - 1)) // max(count, 1)
    ar = img.width / max(img.height, 1)
    cw = int(round(ch * ar))
    if cw > max_w:                              # wide inputs (DL3DV 448x256) hit the width cap first
        cw, ch = max_w, int(round(max_w / ar))
    return max(cw, 1), max(ch, 1)


def _paste(canvas, img, box, draw):
    x, y, w, h = box
    canvas.paste(img.resize((w, h), Image.LANCZOS), (x, y))
    draw.rectangle([x, y, x + w - 1, y + h - 1], outline=COMPOSE_BORDER, width=1)


def compose_recording(shots: list, ctx: list, preds: list) -> list:
    """Lay each captured cloud frame (RGBA) out with the context inputs and the matching
    (color, depth) prediction pair `preds[k]` for the view revealed at step k."""
    from PIL import ImageDraw
    W, H = shots[0].size
    gap = max(6, H // 80)
    pad = max(12, H // 48)
    fs = max(13, H // 26)
    font = _label_font(fs)
    label_h = fs + max(8, fs // 2)

    lw, lh = _stack_cell(ctx[0], len(ctx), H, int(0.40 * W), gap) if ctx else (0, 0)
    pex = next((p for p in preds if p), None)
    rw, rh = _stack_cell(pex[0], 2, H, int(0.52 * W), gap) if pex else (0, 0)

    lx = pad
    cx = lx + (lw + pad if lw else 0)
    rx = cx + W + (pad if rw else 0)
    total = (rx + rw + pad, pad + label_h + H + pad)
    top = pad + label_h

    def col_y(cell_h, n):   # columns centered against the cloud panel, not top-aligned
        return top + max((H - (cell_h * n + gap * (n - 1))) // 2, 0)

    tpl = Image.new("RGB", total, COMPOSE_BG)
    d = ImageDraw.Draw(tpl)

    def caption(text, x, w):
        if w <= 0:
            return
        tw = d.textbbox((0, 0), text, font=font)[2]
        d.text((x + (w - tw) // 2, pad), text, font=font, fill=COMPOSE_LABEL)

    caption("Context", lx, lw)
    caption("PCD", cx, W)
    caption("Predictions", rx, rw)

    if ctx:
        y = col_y(lh, len(ctx))
        for c in ctx:
            _paste(tpl, c, (lx, y, lw, lh), d)
            y += lh + gap

    ry = col_y(rh, 2)
    out = []
    for shot, pair in zip(shots, preds):
        f = tpl.copy()
        fd = ImageDraw.Draw(f)
        f.paste(Image.alpha_composite(   # alpha_composite avoids dark fringes on point edges
            Image.new("RGBA", (W, H), COMPOSE_BG + (255,)), shot).convert("RGB"), (cx, top))
        if pair and rw:
            for i, img in enumerate(pair):
                _paste(f, img, (rx, ry + i * (rh + gap), rw, rh), fd)
        out.append(f)
    return out


def build_animation(frames: list, path: Path, fps: int, hold: int, fmt: str) -> dict:
    """Assemble captured canvas frames into a looping GIF or an MP4. `hold` repeats the final frame
    so the loop pauses on the completed reconstruction. GIF uses one palette derived from the last
    frame (the fullest one), so the palette does not shift and flicker as the cloud grows.
    """
    frames = frames + [frames[-1]] * max(hold, 0)
    if fmt == "mp4":
        import imageio.v2 as imageio      # optional; GIF needs nothing beyond PIL
        imageio.mimsave(path, [np.asarray(f.convert("RGB")) for f in frames],
                        fps=fps, macro_block_size=1, quality=8)   # macro_block_size=1: no auto-padding
    else:
        # Reserve palette index 0 for the exact background, so it never gets dithered.
        adaptive = frames[-1].convert("RGB").convert(
            "P", palette=Image.ADAPTIVE, colors=255).getpalette()[: 255 * 3]
        pal = Image.new("P", (16, 16))
        pal.putdata(bytes(range(256)))
        pal.putpalette(list(COMPOSE_BG) + adaptive)
        # Pillow's nearest-color lookup can still miss pure white, so re-stamp the background
        # pixels to index 0 afterwards.
        bg = np.array(COMPOSE_BG, dtype=np.uint8)
        quant = []
        for f in frames:
            rgb = f.convert("RGB")
            q = np.asarray(rgb.quantize(palette=pal, dither=Image.FLOYDSTEINBERG)).copy()
            q[(np.asarray(rgb) == bg).all(-1)] = 0
            qi = Image.fromarray(q, "P")
            qi.putpalette(pal.getpalette())
            quant.append(qi)
        # optimize=True merges the repeated hold frames into one long-duration frame.
        quant[0].save(path, save_all=True, append_images=quant[1:],
                      duration=max(int(round(1000 / max(fps, 1))), 20), loop=0, optimize=True)
    n_written = len(frames)
    if fmt == "gif":
        with Image.open(path) as g:
            n_written = g.n_frames
    return {"n_frames": n_written, "seconds": round(len(frames) / max(fps, 1), 2),
            "kb": round(path.stat().st_size / 1024)}


def write_transparent_pdf(img: "Image.Image", path: Path) -> None:
    """Write an RGBA image as a single-page PDF with a transparent background, built by hand so no
    extra dependency (e.g. reportlab) is needed. The alpha channel goes in as an /SMask image next
    to the RGB one, and the page declares a /Group << /S /Transparency >> so viewers respect it.
    """
    img = img.convert("RGBA")
    w, h = img.size
    rgb = zlib.compress(img.convert("RGB").tobytes(), 6)
    alpha = zlib.compress(img.getchannel("A").tobytes(), 6)

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = {}

    def add(num: int, body: bytes, stream: bytes | None = None):
        offsets[num] = len(out)
        out.extend(f"{num} 0 obj\n".encode())
        out.extend(body)
        if stream is not None:
            out.extend(b"\nstream\n")
            out.extend(stream)
            out.extend(b"\nendstream")
        out.extend(b"\nendobj\n")

    add(1, b"<< /Type /Catalog /Pages 2 0 R >>")
    add(2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    add(3, (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w} {h}] "
            f"/Group << /S /Transparency /CS /DeviceRGB >> "
            f"/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>").encode())
    add(4, (f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace /DeviceRGB "
            f"/BitsPerComponent 8 /Filter /FlateDecode /SMask 6 0 R /Length {len(rgb)} >>").encode(),
        bytes(rgb))
    content = f"q {w} 0 0 {h} 0 0 cm /Im0 Do Q".encode()   # maps the image onto the whole page
    add(5, f"<< /Length {len(content)} >>".encode(), content)
    add(6, (f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace /DeviceGray "
            f"/BitsPerComponent 8 /Filter /FlateDecode /Length {len(alpha)} >>").encode(),
        bytes(alpha))

    xref = len(out)
    out.extend(f"xref\n0 {len(offsets) + 1}\n".encode())
    out.extend(b"0000000000 65535 f \n")
    for num in sorted(offsets):
        out.extend(f"{offsets[num]:010d} 00000 n \n".encode())
    out.extend((f"trailer\n<< /Size {len(offsets) + 1} /Root 1 0 R >>\n"
                f"startxref\n{xref}\n%%EOF\n").encode())
    path.write_bytes(bytes(out))


class ViewerState:
    def __init__(self, datasets, snapshots: Path):
        self.datasets = datasets                    # {key: Path}
        self.snapshots = snapshots
        self.lock = threading.Lock()
        self.cache = OrderedDict()
        self.meta_cache = {}
        self.scenes = {k: list_scenes(v) for k, v in datasets.items()}
        for k, v in self.scenes.items():
            print(f"[viewer] {k}: {len(v)} scenes under {datasets[k]}")

    def scene_dir(self, dataset, scene) -> Path:
        return self.datasets[dataset] / scene

    def meta(self, dataset, scene):
        key = (dataset, scene)
        if key not in self.meta_cache:
            self.meta_cache[key] = load_scene_meta(self.scene_dir(dataset, scene))
        return self.meta_cache[key]

    def cloud(self, dataset, scene, stride):
        key = (dataset, scene, stride)
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
        payload = build_cloud(self.scene_dir(dataset, scene), self.meta(dataset, scene), stride)
        with self.lock:
            self.cache[key] = payload
            self.cache.move_to_end(key)
            while len(self.cache) > CLOUD_CACHE_SIZE:
                self.cache.popitem(last=False)
        return payload


class Handler(SimpleHTTPRequestHandler):
    state: ViewerState = None

    def log_message(self, fmt, *args):
        pass  # the default per-request stderr spam makes the console unusable

    def _send(self, body: bytes, content_type: str, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status=200):
        self._send(json.dumps(obj).encode(), "application/json", status)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                return self._send(INDEX_HTML.read_bytes(), "text/html; charset=utf-8")

            if u.path == "/api/index":
                # has_fit is re-checked on every request, so the list stays live while a
                # background export is still writing scale_fit.json files.
                scenes = [
                    {"dataset": d, "scene": s,
                     "has_fit": (self.state.datasets[d] / s / "scale_fit.json").exists()}
                    for d, lst in self.state.scenes.items() for s in lst
                ]
                return self._json({"datasets": list(self.state.datasets), "scenes": scenes})

            if u.path == "/api/scene":
                m = dict(self.state.meta(q["dataset"], q["scene"]))
                # extrinsics/intrinsics are tensors in the cache (for build_cloud); make them JSON-safe.
                m["extrinsics"] = m["extrinsics"].tolist() if torch.is_tensor(m["extrinsics"]) else m["extrinsics"]
                m["intrinsics"] = m["intrinsics"].tolist() if torch.is_tensor(m["intrinsics"]) else m["intrinsics"]
                return self._json(m)

            if u.path == "/api/cloud":
                payload, info = self.state.cloud(q["dataset"], q["scene"], int(q.get("stride", 3)))
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("X-Cloud-Info", json.dumps(info))
                self.end_headers()
                return self.wfile.write(payload)

            if u.path == "/api/image":
                d = self.state.scene_dir(q["dataset"], q["scene"])
                kind, frame = q.get("kind", "color"), int(q["frame"])
                p = d / kind / f"{frame:0>6}.png"
                if not p.exists():
                    return self._json({"error": f"missing {p.name}"}, 404)
                return self._send(p.read_bytes(), "image/png")

            return self._json({"error": "not found"}, 404)
        except Exception as exc:                     # surface errors in the UI, not just the console
            import traceback; traceback.print_exc()
            return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def context_images(self, dataset: str, scene: str) -> list:
        """The model's input views, in order (falls back to color_gt if context_image_* is missing)."""
        sdir = self.state.scene_dir(dataset, scene)
        out = []
        for idx in self.state.meta(dataset, scene)["context_index"]:
            for cand in (sdir / f"context_image_{int(idx):0>6}.png",
                         sdir / "color_gt" / f"{int(idx):0>6}.png"):
                if cand.exists():
                    out.append(Image.open(cand).convert("RGB"))
                    break
        return out

    def prediction_frames(self, dataset: str, scene: str, order: list) -> list:
        """(color, depth) prediction pair per recorded step, or None if either file is missing."""
        sdir = self.state.scene_dir(dataset, scene)
        tgt = self.state.meta(dataset, scene)["target_index"]
        out = []
        for pos in order:
            if not 0 <= pos < len(tgt):
                out.append(None)
                continue
            c, g = (sdir / k / f"{int(tgt[pos]):0>6}.png" for k in ("color", "depth"))
            out.append((Image.open(c).convert("RGB"), Image.open(g).convert("RGB"))
                       if c.exists() and g.exists() else None)
        return out

    def do_POST(self):
        u = urlparse(self.path)
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if u.path == "/api/snapshot":
            raw = base64.b64decode(body["png"].split(",", 1)[1])
            img = Image.open(io.BytesIO(raw)).convert("RGBA")
            self.state.snapshots.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            base = f"{body.get('dataset','x')}_{body.get('scene','x')[:16]}_{stamp}"
            png_path = self.state.snapshots / f"{base}.png"
            pdf_path = self.state.snapshots / f"{base}.pdf"
            img.save(png_path)
            write_transparent_pdf(img, pdf_path)
            # Save the viewer state alongside, so the figure can be reproduced later.
            (self.state.snapshots / f"{base}.json").write_text(json.dumps(body.get("state", {}), indent=2))

            # Save the context views (model inputs) next to the render too.
            ctx_saved = []
            try:
                sdir = self.state.scene_dir(body["dataset"], body["scene"])
                for n, idx in enumerate(self.state.meta(body["dataset"], body["scene"])["context_index"]):
                    for cand in (sdir / f"context_image_{int(idx):0>6}.png",
                                 sdir / "color_gt" / f"{int(idx):0>6}.png"):
                        if cand.exists():
                            dst = self.state.snapshots / f"{base}_context{n}_{int(idx):0>6}.png"
                            dst.write_bytes(cand.read_bytes())
                            ctx_saved.append(dst.name)
                            break
            except Exception as exc:            # never lose the snapshot over a missing input image
                print(f"[viewer] context views not copied: {type(exc).__name__}: {exc}")

            return self._json({"ok": True, "pdf": str(pdf_path), "png": str(png_path),
                               "size": f"{img.size[0]}x{img.size[1]}",
                               "kb": round(pdf_path.stat().st_size / 1024),
                               "context": ctx_saved})
        if u.path == "/api/record":
            fmt = "mp4" if body.get("format") == "mp4" else "gif"
            if fmt == "mp4":
                try:
                    import imageio.v2  # noqa: F401
                except ImportError:
                    return self._json({"error": "mp4 needs imageio + imageio-ffmpeg; use GIF"}, 400)
            imgs = [Image.open(io.BytesIO(base64.b64decode(f.split(",", 1)[1]))).convert("RGBA")
                    for f in body["frames"]]
            if not imgs:
                return self._json({"error": "no frames received"}, 400)

            dataset, scene = body.get("dataset", "x"), body.get("scene", "x")
            order = [int(i) for i in body.get("order", [])]
            note = ""
            if body.get("compose", True):
                try:
                    imgs = compose_recording(imgs, self.context_images(dataset, scene),
                                             self.prediction_frames(dataset, scene, order))
                except Exception as exc:
                    note = f"composite skipped: {type(exc).__name__}: {exc}"
                    print(f"[viewer] {note}")
            if imgs[0].mode == "RGBA":          # plain recording still gets the white ground
                imgs = [Image.alpha_composite(Image.new("RGBA", i.size, (255, 255, 255, 255)), i)
                        for i in imgs]

            out_dir = self.state.snapshots / "gifs"
            out_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            base = f"{dataset}_{scene[:16]}_{stamp}_recon"
            out = out_dir / f"{base}.{fmt}"
            info = build_animation([i.convert("RGB") for i in imgs], out,
                                   int(body.get("fps", 12)), int(body.get("hold", 0)), fmt)
            (out_dir / f"{base}.json").write_text(json.dumps(body.get("state", {}), indent=2))
            return self._json({"ok": True, "path": str(out),
                               "size": f"{imgs[0].size[0]}x{imgs[0].size[1]}",
                               "note": note, **info})
        return self._json({"error": "not found"}, 404)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", action="append", required=True, metavar="KEY=PATH",
                        help="Repeatable, e.g. --dataset re10k=outputs/video_eval_re10k_...")
    parser.add_argument("--snapshots", default="assets/pcd_snapshots",
                        help="Where 'save snapshot' writes the transparent PDF/PNG plus a JSON of "
                             "the viewer state that produced it")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    datasets = {}
    for spec in args.dataset:
        if "=" not in spec:
            raise SystemExit(f"--dataset expects KEY=PATH, got {spec!r}")
        k, p = spec.split("=", 1)
        datasets[k] = Path(p)

    Handler.state = ViewerState(datasets, Path(args.snapshots))
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[viewer] serving on http://{args.host}:{args.port}  (Ctrl+C to stop)")
    print(f"[viewer] snapshots -> {args.snapshots}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[viewer] stopped")


if __name__ == "__main__":
    main()
