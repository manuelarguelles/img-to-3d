#!/usr/bin/env python3
"""img-to-3d: Image → TRELLIS → GLB/STL → f3d viewer"""

import sys
import os
import shutil
import argparse
import subprocess
import tempfile
from pathlib import Path
from gradio_client import Client, handle_file

OUTPUT_DIR = Path.home() / "Downloads" / "img-to-3d"
HF_SPACE = "trellis-community/trellis"

# TRELLIS preprocess + generation defaults
PREPROCESS_BACKGROUND = True
SEED = 1
SS_GUIDANCE = 7.5
SS_STEPS = 12
SLAT_GUIDANCE = 3.0
SLAT_STEPS = 12
MESH_SIMPLIFY = 0.95
TEXTURE_SIZE = 1024


def preprocess(client: Client, image_path: str) -> str:
    result = client.predict(
        image=handle_file(image_path),
        prepro_ckpt="none",
        background_choice="Auto Remove Background" if PREPROCESS_BACKGROUND else "Original Image",
        foreground_ratio=0.85,
        seed=SEED,
        randomize_seed=False,
        api_name="/preprocess_image",
    )
    # result may be a path or (path, seed) tuple
    if isinstance(result, (list, tuple)):
        preprocessed_path = result[0]
        seed_used = result[1] if len(result) > 1 else SEED
    else:
        preprocessed_path = result
        seed_used = SEED
    return preprocessed_path, seed_used


def generate(client: Client, preprocessed_path: str, seed: int) -> str:
    result = client.predict(
        image=handle_file(preprocessed_path),
        seed=seed,
        randomize_seed=False,
        ss_guidance_strength=SS_GUIDANCE,
        ss_sampling_steps=SS_STEPS,
        slat_guidance_strength=SLAT_GUIDANCE,
        slat_sampling_steps=SLAT_STEPS,
        multiimage_algo="stochastic",
        api_name="/image_to_3d",
    )
    return result  # returns video preview + 3D state


def extract_glb(client: Client, state) -> Path:
    result = client.predict(
        mesh_simplify=MESH_SIMPLIFY,
        texture_size=TEXTURE_SIZE,
        api_name="/extract_glb",
    )
    if isinstance(result, (list, tuple)):
        glb_path = result[0]
    else:
        glb_path = result
    return Path(glb_path)


def to_stl(glb_path: Path, stl_path: Path):
    try:
        import trimesh
        mesh = trimesh.load(str(glb_path), force="mesh")
        mesh.export(str(stl_path))
        print(f"  STL: {stl_path.stat().st_size // 1024} KB  |  faces: {len(mesh.faces)}")
    except ImportError:
        print("  [aviso] trimesh no instalado — solo se genera el GLB")


def open_f3d(path: Path):
    f3d = shutil.which("f3d")
    if not f3d:
        print("  f3d no encontrado. Instalar con: brew install f3d")
        return
    subprocess.Popen([f3d, str(path)])
    print(f"  Abriendo en f3d: {path.name}")


def run(image_path: str, output_name: str | None, stl: bool, view: bool):
    image_path = os.path.expanduser(image_path)
    if not os.path.exists(image_path):
        print(f"Error: no se encuentra la imagen: {image_path}")
        sys.exit(1)

    stem = output_name or Path(image_path).stem
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    glb_out = OUTPUT_DIR / f"{stem}.glb"
    stl_out = OUTPUT_DIR / f"{stem}.stl"

    print(f"[1/4] Conectando a TRELLIS ({HF_SPACE})...")
    client = Client(HF_SPACE)

    print("[2/4] Preprocesando imagen...")
    preprocessed, seed = preprocess(client, image_path)

    print("[3/4] Generando modelo 3D...")
    state = generate(client, preprocessed, seed)

    print("[4/4] Extrayendo GLB...")
    tmp_glb = extract_glb(client, state)
    shutil.copy(tmp_glb, glb_out)
    print(f"  GLB: {glb_out}  ({glb_out.stat().st_size // 1024} KB)")

    view_target = glb_out
    if stl:
        to_stl(glb_out, stl_out)
        view_target = stl_out

    if view:
        open_f3d(view_target)

    print(f"\nListo. Archivos en {OUTPUT_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Imagen → TRELLIS → 3D → f3d")
    parser.add_argument("image", help="Ruta a la imagen fuente")
    parser.add_argument("-n", "--name", help="Nombre base del archivo de salida")
    parser.add_argument("--stl", action="store_true", help="Exportar también a STL")
    parser.add_argument("--no-view", action="store_true", help="No abrir en f3d")
    args = parser.parse_args()
    run(args.image, args.name, args.stl, not args.no_view)


if __name__ == "__main__":
    main()
