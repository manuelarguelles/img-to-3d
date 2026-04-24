#!/usr/bin/env python3
"""img-to-3d: Image → TRELLIS (Replicate) → GLB + STL → f3d viewer

Modes:
  Batch: python img_to_3d.py          → procesa todo /raw, salta ya procesados
  Single: python img_to_3d.py <image> → procesa una imagen específica
"""

import sys
import os
import shutil
import argparse
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

REPO_DIR      = Path(__file__).parent
RAW_DIR       = REPO_DIR / "raw"
OUTPUT_DIR    = REPO_DIR / "output"
REPLICATE_MODEL = "firtoz/trellis:e8f6c45206993f297372f5436b90350817bd9b4a0d52d2a76df50c1c8afa2b3c"
IMAGE_EXTS    = {".jpg", ".jpeg", ".png", ".webp"}

SS_GUIDANCE   = 7.5
SS_STEPS      = 12
SLAT_GUIDANCE = 3.0
SLAT_STEPS    = 12
MESH_SIMPLIFY = 0.95
TEXTURE_SIZE  = 1024

F3D_CANDIDATES = [
    shutil.which("f3d"),
    "/opt/homebrew/bin/f3d",
    "/opt/homebrew/var/homebrew/tmp/.cellar/f3d/3.5.0/bin/f3d",
    "/usr/local/bin/f3d",
]


def find_f3d():
    return next((p for p in F3D_CANDIDATES if p and os.path.isfile(p)), None)


def already_processed(stem: str) -> bool:
    return (OUTPUT_DIR / f"{stem}.stl").exists()


def generate_glb(image_path: Path) -> bytes:
    import replicate
    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        print("Error: REPLICATE_API_TOKEN no encontrado en .env")
        sys.exit(1)
    client = replicate.Client(api_token=token)
    with open(image_path, "rb") as f:
        output = client.run(
            REPLICATE_MODEL,
            input={
                "images": [f],
                "texture_size": TEXTURE_SIZE,
                "mesh_simplify": MESH_SIMPLIFY,
                "generate_model": True,
                "generate_color": True,
                "randomize_seed": False,
                "seed": 1,
                "ss_guidance_strength": SS_GUIDANCE,
                "ss_sampling_steps": SS_STEPS,
                "slat_guidance_strength": SLAT_GUIDANCE,
                "slat_sampling_steps": SLAT_STEPS,
            },
        )
    return output["model_file"].read()


def glb_to_stl(glb_path: Path, stl_path: Path):
    import trimesh
    mesh = trimesh.load(str(glb_path), force="mesh")
    mesh.export(str(stl_path))
    print(f"  STL: {stl_path.name}  ({stl_path.stat().st_size // 1024} KB, {len(mesh.faces)} caras)")


def open_f3d(path: Path):
    f3d = find_f3d()
    if not f3d:
        print("  f3d no encontrado — instalar con: brew install f3d")
        return
    subprocess.Popen([f3d, str(path)])
    print(f"  Abriendo en f3d: {path.name}")


def process_image(image_path: Path, view: bool = True):
    stem = image_path.stem
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    glb_out = OUTPUT_DIR / f"{stem}.glb"
    stl_out = OUTPUT_DIR / f"{stem}.stl"

    print(f"\n→ {image_path.name}")

    if already_processed(stem):
        print(f"  [skip] {stem}.stl ya existe en /output")
        return stl_out

    print("  [1/3] Enviando a Replicate TRELLIS...")
    glb_bytes = generate_glb(image_path)

    print("  [2/3] Guardando GLB...")
    glb_out.write_bytes(glb_bytes)
    print(f"  GLB: {glb_out.name}  ({glb_out.stat().st_size // 1024} KB)")

    print("  [3/3] Convirtiendo a STL...")
    glb_to_stl(glb_out, stl_out)

    if view:
        open_f3d(stl_out)

    return stl_out


def batch(view: bool):
    images = sorted(p for p in RAW_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"No hay imágenes en {RAW_DIR}")
        return

    pending = [img for img in images if not already_processed(img.stem)]
    skipped = len(images) - len(pending)

    print(f"Encontradas: {len(images)} imágenes  |  pendientes: {len(pending)}  |  ya procesadas: {skipped}")

    for img in pending:
        process_image(img, view=view)

    print(f"\nListo. Archivos en {OUTPUT_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Imagen → TRELLIS → GLB + STL → f3d")
    parser.add_argument("image", nargs="?", help="Imagen a procesar (omitir = batch desde /raw)")
    parser.add_argument("--no-view", action="store_true", help="No abrir en f3d")
    args = parser.parse_args()

    view = not args.no_view

    if args.image:
        image_path = Path(os.path.expanduser(args.image))
        if not image_path.exists():
            print(f"Error: no se encuentra {image_path}")
            sys.exit(1)
        process_image(image_path, view=view)
    else:
        batch(view=view)


if __name__ == "__main__":
    main()
