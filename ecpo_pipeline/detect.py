from ecpo_pipeline.utils import discover_images
from PIL import Image, ImageDraw
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

import asyncio
import itertools
import math
import networkx
import numpy as np
import paddleocr
import pathlib
import tqdm


# The threshold to use to decide whether two boxes are disjoint.
# If the intersection of the two boxes is smaller than this threshold
# multiplied with the area of the smaller box, they are considered disjoint.
THRESHOLD_DISJOINT = 0.1

# The threshold to use to decide whether boxes cover the full image.
# The union of the boxes need to be at least this threshold
# multiplied with the area of the image.
THRESHOLD_COVERS_FULL = 0.85

# The confidence threshold for text detection. For complex layouts,
# PaddleOCR will give very low confidence scores, so we set this very low.
# For text-heavy "easy" layouts, we should raise it, since we otherwise
# get to many overlapping boxes for our algorithms to handle.
THRESHOLD_TEXT_CONFIDENCE = 0.1

# The threshold for image confidence scores. Not sure yet in which cases
# to change it. Too low values lead to an image box being overlayed over
# the entire image.
THRESHOLD_IMAGE_CONFIDENCE = 0.25

_bar_layout = None
_semaphore = asyncio.Semaphore(30)


def poor_mans_defaultdict(default, args):
    """Workaround for PaddleOCR's inability to use collections.defaultdict

    This should be a default dict, but PaddleOCR does not use the passed
    dictionary according to its API, and therefore does not instantiate the defaults."""
    result = {i: default for i in range(25)}
    result.update(args)
    return result


detector = paddleocr.LayoutDetection(
    threshold=poor_mans_defaultdict(
        1.0,
        {
            0: THRESHOLD_TEXT_CONFIDENCE,
            1: THRESHOLD_IMAGE_CONFIDENCE,
            2: THRESHOLD_TEXT_CONFIDENCE,
        },
    ),
    layout_merge_bboxes_mode="union",
)


def disjoint_groups(items, is_disjoint):
    """
    items: iterable of items
    is_disjoint(a, b): returns True if a and b are disjoint (i.e., should NOT be in same group)
    returns: list of sets, each set contains items that are NOT disjoint from each other
    """
    uf = networkx.utils.UnionFind(items)

    # Merge pairs that are NOT disjoint
    for a, b in itertools.combinations(items, 2):
        if not is_disjoint(a, b):
            uf.union(a, b)

    return list(uf.to_sets())


# def crop_polygon(image: np.ndarray, polygon: Polygon) -> np.ndarray:
#     # Create mask image
#     mask = Image.new("L", (image.shape[1], image.shape[0]), 0)
#     draw = ImageDraw.Draw(mask)

#     # Handle Polygon and MultiPolygon
#     if isinstance(polygon, Polygon):
#         polygons = [polygon]
#     elif isinstance(polygon, MultiPolygon):
#         polygons = list(polygon.geoms)
#     else:
#         raise TypeError("Input must be a Polygon or MultiPolygon")

#     # Draw all polygons on the mask
#     for poly in polygons:
#         poly_coords = list(poly.exterior.coords)
#         draw.polygon(poly_coords, outline=1, fill=1)

#     mask = np.array(mask)

#     # Apply mask on each channel
#     cropped = image.copy()
#     if len(image.shape) == 3:  # RGB or similar
#         cropped[mask == 0] = 0
#     else:  # Grayscale
#         cropped = cropped * mask

#     # Auto-crop to bounding box of polygon
#     minx, miny, maxx, maxy = polygon.bounds
#     minx, miny, maxx, maxy = map(int, [minx, miny, maxx, maxy])

#     return cropped[miny:maxy, minx:maxx]


def rasterize_polygon_to_mask(image_shape, polygon):
    """
    Rasterize a shapely Polygon or MultiPolygon to a boolean mask of shape (H, W).
    Holes are handled. Coordinates are rounded to integer pixel coordinates
    with a consistent floor/ceil strategy.
    """
    H, W = image_shape[0], image_shape[1]
    if isinstance(polygon, Polygon):
        polygons = [polygon]
    elif isinstance(polygon, MultiPolygon):
        polygons = list(polygon.geoms)
    else:
        raise TypeError("polygon must be shapely Polygon or MultiPolygon")

    # Create mask image (L mode gives 0..255 values)
    mask_img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask_img)

    for poly in polygons:
        # Exterior (rounded to nearest integer). We use rounding to nearest pixel.
        exterior_coords = [
            (int(round(x)), int(round(y))) for x, y in poly.exterior.coords
        ]
        draw.polygon(exterior_coords, outline=255, fill=255)

        # Interiors -> holes: draw them with fill=0 to erase
        for interior in poly.interiors:
            interior_coords = [
                (int(round(x)), int(round(y))) for x, y in interior.coords
            ]
            draw.polygon(interior_coords, outline=0, fill=0)

    mask = np.array(mask_img, dtype=np.uint8)  # 0 or 255
    mask_bool = mask != 0  # boolean mask: True inside polygon
    return mask_bool


def crop_polygon(image: np.ndarray, polygon):
    """
    Returns (cropped_image, cropped_mask).
    cropped_mask is boolean (True = inside polygon).
    The cropping bbox is computed from polygon.bounds and clamped to image.
    """
    H, W = image.shape[0], image.shape[1]
    mask = rasterize_polygon_to_mask((H, W), polygon)

    # Compute integer bbox: floor(min), ceil(max) and clamp
    minx, miny, maxx, maxy = polygon.bounds
    minx = max(int(math.floor(minx)), 0)
    miny = max(int(math.floor(miny)), 0)
    maxx = min(int(math.ceil(maxx)), W)
    maxy = min(int(math.ceil(maxy)), H)

    # Crop both image and mask
    mask_cropped = mask[miny:maxy, minx:maxx]

    # Prepare cropped image: zero outside polygon in the bbox
    if image.ndim == 3:
        cropped_img = image[miny:maxy, minx:maxx].copy()
        # Broadcast mask to channels
        mask_3c = np.repeat(
            mask_cropped[:, :, np.newaxis], cropped_img.shape[2], axis=2
        )
        cropped_img[~mask_3c] = 255
    else:
        cropped_img = image[miny:maxy, minx:maxx].copy()
        cropped_img[~mask_cropped] = 255

    return cropped_img, mask_cropped


def extract_offset(bbox):
    x0, y0, _, _ = map(int, bbox)
    return x0, y0


def maximal_cliques(items, rel):
    G = networkx.Graph()
    G.add_nodes_from(items)
    for i in items:
        for j in items:
            if i != j:
                if rel(i, j):
                    G.add_edge(i, j)

    return list(networkx.find_cliques(G))


def box_to_polygon(x0, y0, x1, y1):
    """Convert a bounding box to a Shapely Polygon."""
    return Polygon(
        [
            (x0, y0),  # top-left
            (x1, y0),  # top-right
            (x1, y1),  # bottom-right
            (x0, y1),  # bottom-left
        ]
    )


def subtract_images(text_polys, image_polys):
    all_images = unary_union(image_polys)

    result = []
    for tp in text_polys:
        old_area = tp.area
        tp = tp.difference(all_images)
        new_area = tp.area

        if new_area / old_area > 0.1:
            result.append(tp)

    return result


def disjoint_criterion(p1, p2):
    return p1.intersection(p2).area < THRESHOLD_DISJOINT * min(p1.area, p2.area)


def filter_union_boxes(polys, fullsize):
    # print(f"fullsize {fullsize}")
    cliques = maximal_cliques(polys, disjoint_criterion)

    # Sort cliques by total sizes, then by clique size.
    for clique in reversed(
        sorted(
            sorted(cliques, key=lambda clique: unary_union(clique).area),
            key=lambda c: len(c),
        )
    ):
        # print(f"clique area {unary_union(clique).area}, length {len(clique)}")
        if unary_union(clique).area > THRESHOLD_COVERS_FULL * fullsize:
            # print("RETURNING")
            return clique

    # If we did not find a clique that covers everything, we might need to split
    # this first. We do so by checking whether removing a single box does split the
    # polygon into two separate ones. Then, to validate that we are not discarding
    # relevant context, we double checkt that the polygon does not shrink substantially
    # by removing the polygon. An improved version could go ahead and check how much
    # "black" is found in what we discard.

    # If we make it here, we did not find something meaningful, we merge all boxes and hope for the best.
    print("We created a union!")
    return [unary_union(clique)]


async def analyse_image(imagefile: pathlib.Path):
    async with _semaphore:
        image = np.array(Image.open(imagefile))

        # Detect all boxes, even when they massively overlap
        layout = detector.predict(image)
        boxes = layout[0]["boxes"]

        # Split text and image boxes and convert them to polygons
        text_polys = [
            box_to_polygon(*b["coordinate"]) for b in boxes if b["cls_id"] in (0, 2)
        ]
        image_polys = [
            box_to_polygon(*b["coordinate"]) for b in boxes if b["cls_id"] in (1,)
        ]

        # Images are typically accurate, so we cut them from the text polys now
        text_polys = subtract_images(text_polys, image_polys)

        # Revise text box groups by finding a maximum cover with
        # small boxes. This is our custom heuristic, because Paddle
        # itself will either generate too big boxes or miss space when
        # using small boxes.
        poly_groups = disjoint_groups(text_polys, disjoint_criterion)
        revised_polys = []
        for boxg in poly_groups:
            group_boxes = filter_union_boxes(boxg, unary_union(list(boxg)).area)
            revised_polys.extend(group_boxes)

        _bar_layout.update(1)

        return {
            "imagefile": str(imagefile),
            "text_polys": revised_polys,
            "image_polys": image_polys,
        }


async def analyse_layout(images: list[pathlib.Path]):
    image_tasks = {i.stem: asyncio.create_task(analyse_image(i)) for i in images}

    global _bar_layout
    _bar_layout = tqdm.tqdm(total=len(image_tasks), desc="Analysing layout", position=0)

    await asyncio.gather(*image_tasks.values())

    results = {}
    for stem, it in image_tasks.items():
        results[stem] = it.result()

    return results


def overlay(image, result):
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")

    def _draw_polygons(polys, color):
        for poly in polys:
            if isinstance(poly, MultiPolygon):
                poly_iter = poly.geoms
            else:
                poly_iter = [poly]

            for p in poly_iter:
                draw.polygon(p.exterior.coords, fill=color)

    _draw_polygons(result["image_polys"], (60, 180, 75, 128))
    _draw_polygons(result["text_polys"], (230, 25, 75, 128))

    return image


if __name__ == "__main__":
    result = asyncio.run(
        analyse_layout(
            discover_images(
                pathlib.Path(
                    "/home/dkempf/heibox/ECPO_data/images_with_groundtruth/png/jingbao/1920/04"
                )
            )
        )
    )

    pathlib.Path("output_layout").mkdir(exist_ok=True)

    for stem, image in tqdm.tqdm(result.items(), desc="Saving layout overlays"):
        overlay_img = overlay(image["imagefile"], image["boxes"])
        overlay_img.save(pathlib.Path("output_layout") / f"{stem}_layout.png")
