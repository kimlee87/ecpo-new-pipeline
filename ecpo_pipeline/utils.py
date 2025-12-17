import pathlib


def discover_images(image_path: pathlib.Path):
    return (
        list(image_path.rglob("*.png"))
        + list(image_path.rglob("*.jpg"))
        + list(image_path.rglob("*.tiff"))
    )
