# img-to-3d

Pipeline completo: imagen → modelo 3D → visualización local con f3d.

Usa [TRELLIS](https://huggingface.co/spaces/trellis-community/trellis) vía Gradio API de Hugging Face.

## Requisitos

```bash
brew install f3d
pip install -r requirements.txt
```

## Uso

```bash
# Generar GLB y abrir en f3d
python img_to_3d.py ~/Downloads/bulbasaur.png

# Generar GLB + STL y abrir en f3d
python img_to_3d.py ~/Downloads/bulbasaur.png --stl

# Nombre personalizado de salida
python img_to_3d.py ~/Downloads/figura.jpg -n mi_figura --stl

# Solo generar, sin abrir visor
python img_to_3d.py imagen.png --no-view
```

## Salida

Los archivos se guardan en `~/Downloads/img-to-3d/`:
- `<nombre>.glb` — modelo con texturas PBR (abre en f3d, Blender, etc.)
- `<nombre>.stl` — geometría pura para impresión 3D (si usas `--stl`)

## Flujo interno

```
imagen → preprocess_image (remove bg) → image_to_3d (TRELLIS) → extract_glb → [trimesh STL] → f3d
```

## Skill Claude Code

Agrega el directorio al PATH o crea un alias:

```bash
alias img3d="python ~/clawd/img-to-3d/img_to_3d.py"
```
