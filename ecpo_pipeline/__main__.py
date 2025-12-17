from ecpo_pipeline.detect import analyse_layout, crop_polygon, overlay
from ecpo_pipeline.utils import discover_images
from ecpo_pipeline.vllm import perform_vllm_ocr

from PIL import Image

import asyncio
import click
import json
import numpy as np
import pathlib
import tqdm


# @click.command()
def pipeline():
    layout = asyncio.run(
        analyse_layout(
            discover_images(pathlib.Path("/home/dkempf/ecpo-new-pipeline/input"))
        )
    )

    pathlib.Path("output_layout").mkdir(exist_ok=True)

    for stem, image in tqdm.tqdm(layout.items(), desc="Saving layout overlays"):
        overlay_img = overlay(image["imagefile"], image)
        overlay_img.save(pathlib.Path("output_layout") / f"{stem}_layout.png")

    pathlib.Path("output_crops").mkdir(exist_ok=True)

    # Try the following only on one
    stem, layout = list(layout.items())[1]
    image = np.array(Image.open(layout["imagefile"]))
    for i, poly in tqdm.tqdm(enumerate(layout["text_polys"]), desc="Storing crops"):
        crop = crop_polygon(image, poly)
        Image.fromarray(crop).save(
            pathlib.Path("output_crops") / f"{str(i).zfill(4)}.png"
        )

    # Run OCR
    result = asyncio.run(perform_vllm_ocr(pathlib.Path("output_crops")))

    with open("results.json", "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    pipeline()
