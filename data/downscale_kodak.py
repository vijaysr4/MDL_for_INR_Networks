import os
from typing import List, Tuple

from PIL import Image
from tqdm import tqdm


IMAGES_DIR: str = "/users/eleves-b/2024/vijay-venkatesh.murugan/.cache/kagglehub/datasets/sherylmehta/kodak-dataset/versions/1"
OUTPUT_DIR: str = "downscaled_imgs"
MAX_SIDE: int = 128  # change this if you want bigger or smaller images


def get_image_paths(images_dir: str) -> List[str]:
    exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    paths: List[str] = []
    for fname in sorted(os.listdir(images_dir)):
        if fname.lower().endswith(exts):
            paths.append(os.path.join(images_dir, fname))
    return paths


def downscale_image(in_path: str, out_path: str, max_side: int) -> None:
    img = Image.open(in_path)
    w, h = img.size
    scale = min(max_side / float(w), max_side / float(h), 1.0)
    if scale < 1.0:
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.BICUBIC)
    img.save(out_path)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image_paths: List[str] = get_image_paths(IMAGES_DIR)
    if not image_paths:
        raise FileNotFoundError(f"No image files found in directory: {IMAGES_DIR}")

    for in_path in tqdm(image_paths, desc="Downscaling images", ncols=100):
        fname = os.path.basename(in_path)
        out_path = os.path.join(OUTPUT_DIR, fname)
        downscale_image(in_path, out_path, MAX_SIDE)

    print(f"Downscaled images saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
