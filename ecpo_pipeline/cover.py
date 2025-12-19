import itertools
import math
import networkx as nx
import numpy as np
import paddleocr

import _cover_heuristic

from PIL import Image, ImageDraw
from shapely.affinity import translate
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union, polygonize
from skimage.filters import threshold_otsu


def poor_mans_defaultdict(default, args):
    """Workaround for PaddleOCR's inability to use collections.defaultdict

    This should be a default dict, but PaddleOCR does not use the passed
    dictionary according to its API, and therefore does not instantiate the defaults."""
    result = {i: default for i in range(25)}
    result.update(args)
    return result


# Global instance of the detector. The parameters do not matter too much, as we
# can always pass them to inference without performance penalties. It is however
# important to have this as a singleton, as it allocates the GPU memory.
detector = paddleocr.LayoutDetection(
    threshold=poor_mans_defaultdict(
        1.0,
        {
            0: 0.01,
            1: 0.25,
            2: 0.01,
        },
    ),
    layout_merge_bboxes_mode="union",
)


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


def otsu_binarization(img):
    """Apply binarization to an image.

    Not refined enough for OCR, but well worth it for layout detection.
    """
    img = np.array(Image.fromarray(img).convert("L"))
    thr = threshold_otsu(img)
    return ((img > thr) * 255).astype(np.uint8)


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


def calculate_atomics(polys):
    """Given a number of polygons, calculate the set of composing atomics.

    A set of atomics for a set of polygons is defined such that every
    polygon is the disjoint union of a subset of the atomics. Allows
    for additive calculations without resorting to geometry calculations
    every time.
    """
    # Create all atomics as polygons
    boundaries = [p.boundary for p in polys if not p.is_empty]
    merged = unary_union(boundaries)
    atomics = list(polygonize(merged))

    # For each atomic, find out which polygons it belongs to
    atomic_covers = []
    for cell in atomics:
        pt = cell.representative_point()
        covered = [i for i, p in enumerate(polys) if p.covers(pt)]
        atomic_covers.append(covered)

    # Invert that mapping: Which atomics compose each polygon
    poly_atomics = [[] for p in polys]
    for i, ac in enumerate(atomic_covers):
        for p in ac:
            poly_atomics[p].append(i)

    return atomics, poly_atomics


def black_content(binary, poly):
    """Count of black pixels in the polygon."""
    if poly.area == 0.0:
        return 0

    cropped_img, mask_cropped, _ = crop_polygon(binary, poly)
    return np.sum(mask_cropped & (cropped_img == 0))


def squaricity(poly):
    """A measure for how square-like a polygon is.

    Values are always in [0, 1] with 1 is an actual square.
    """
    return 16 * poly.area / (poly.length * poly.length)


def average_squaricity_criterion(polys):
    """Sorting criterion for average squaricity for a set of polygons."""

    def _average_squaricity_criterion(indices):
        return sum((squaricity(polys[i]) for i in indices), 0.0) / len(indices)

    return _average_squaricity_criterion


def overlap_threshold_function(polys, threshold=0.9):
    """Overlap percentage thresholding for two polygons.

    Values are always in [0, 1] with 1 for identical polygons.
    """

    def _func(i, j):
        return (
            polys[i].intersection(polys[j]).area / polys[i].union(polys[j]).area
            > threshold
        )

    return _func


def black_overlap_function(binary, polys, threshold=0.98):
    """Overlap percentage of the black pixels of two polygons.

    Values are always in [0, 1] with 1 for all black content in the overlap.
    This would essentially mean that we can pick any polygon without losing
    anything.
    """

    atomics, poly_atomics = calculate_atomics(polys)
    atomics_values = [black_content(binary, a) for a in atomics]

    def _func(i, j):
        seti = set(poly_atomics[i])
        setj = set(poly_atomics[j])
        intersection = seti.intersection(setj)
        union = seti.union(setj)

        return (
            sum((atomics_values[i] for i in intersection), 0)
            / sum((atomics_values[i] for i in union), 0)
            > threshold
        )

    return _func


def exact_disjoint_criterion(p, q):
    """True if two polygons are disjoint"""
    return p.intersection(q).area == 0


def fuzzy_disjoint_criterion(threshold):
    def _func(p, q):
        return p.intersection(q).area < threshold * min(p.area, q.area)

    return _func


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

    return cropped_img, mask_cropped, (minx, miny)


def filter_redundant_polys(polys, criterion):
    """Filter polygons that are almost identical with others."""

    uf = nx.utils.UnionFind(polys)

    for i, p in enumerate(polys):
        for j, q in enumerate(polys):
            if i < j:
                if criterion(i, j):
                    uf.union(p, q)

    return [unary_union(list(s)) for s in uf.to_sets()]


def intersection_edges(polys):
    """Calculate the intersection graph for a set of polygons."""

    edges = []
    for i, p in enumerate(polys):
        for j, q in enumerate(polys):
            if i != j:
                if not exact_disjoint_criterion(p, q):
                    # if not disjoint_criterion(p, q):
                    edges.append((i, j))

    return edges


def disjoint_groups(items, is_disjoint):
    """
    items: iterable of items
    is_disjoint(a, b): returns True if a and b are disjoint (i.e., should NOT be in same group)
    returns: list of sets, each set contains items that are NOT disjoint from each other
    """
    uf = nx.utils.UnionFind(items)

    # Merge pairs that are NOT disjoint
    for a, b in itertools.combinations(items, 2):
        if not is_disjoint(a, b):
            uf.union(a, b)

    return list(uf.to_sets())


def impl_layout_detection(img, text_threshold=0.05):
    # Run the PaddleOCR layout detection
    layout = detector.predict(
        img,
        threshold=poor_mans_defaultdict(
            1.0,
            {
                0: text_threshold,
                1: 0.25,
                2: text_threshold,
            },
        ),
    )
    boxes = layout[0]["boxes"]

    # Separate text and image boxes and convert them to polygons
    text_polys = [
        box_to_polygon(*b["coordinate"]) for b in boxes if b["cls_id"] in (0, 2)
    ]
    image_polys = [
        box_to_polygon(*b["coordinate"]) for b in boxes if b["cls_id"] in (1,)
    ]

    # Subtract all images from the text polygons
    text_polys = subtract_images(text_polys, image_polys)

    # Drop any polygons that do not contain more than 10 black pixels
    text_polys = [p for p in text_polys if black_content(img, p) > 10]

    # Filter polygons that do not add value
    text_polys = filter_redundant_polys(
        text_polys, overlap_threshold_function(text_polys)
    )
    # The following one would be better, but is way too slow right now
    # text_polys = filter_redundant_polys(
    #     text_polys, black_overlap_function(img, text_polys)
    # )

    # This happened in practice.
    # TODO: investigate why this is even possible.
    if len(text_polys) == 0:
        return text_polys, image_polys

    # Look for disjoint groups of text polygons to apply a divide and conquer approach.
    # We have a choice between making this with an exact disjoint criterion or a fuzzy
    # one. So far, I have been switching back and forth.
    # poly_groups = disjoint_groups(text_polys, fuzzy_disjoint_criterion(0.95))
    poly_groups = disjoint_groups(text_polys, exact_disjoint_criterion)

    # We found a trivial split, so we can do divide and conquer
    if len(poly_groups) > 1:
        # Crop the image according to the polygon groups
        crops = [crop_polygon(img, unary_union(list(pg))) for pg in poly_groups]

        # Recursively call this function for each group and combine the results
        results = []
        for cimg, _, (xoff, yoff) in crops:
            for cpoly in impl_layout_detection(cimg)[0]:
                results.append(translate(cpoly, xoff=xoff, yoff=yoff))

        return results, image_polys

    # If we reach this, all polygons were connected and we need to find correct
    # polygons by selecting a subset. However, our algorithm to do so is of exponential
    # complexity, we therefore can only run in up to a certain threshold around
    # 20 polygons. We "fix" this by increasing the threshold value for detection
    # until we drop below this threshold.
    if len(text_polys) > 20:
        print(
            f"Restarting algorithm (found {len(text_polys)} polygons) with threshold {text_threshold + 0.01}"
        )
        return impl_layout_detection(img, text_threshold=text_threshold + 0.01)

    # As a preparation, we build some data structures
    atomics, poly_atomics = calculate_atomics(text_polys)
    atomics_values = [black_content(img, a) for a in atomics]
    edges = intersection_edges(text_polys)

    # Run the C++ brute-force algorithm with decreasing threshold
    def _bruteforce(threshold):
        if threshold < 0.75:
            # If we reduce the threshold, so drastically, something is terribly wrong.
            # We should investigate this, but for now, I just return the union of polygons
            # as one.
            return [list(range(len(text_polys)))]

        res = _cover_heuristic.find_optimal_cover(
            threshold, edges, poly_atomics, atomics_values
        )
        if len(res) == 0:
            print(f"Restarting brute-forcing with threshold {threshold - 0.01}")
            return _bruteforce(threshold - 0.01)
        return res

    res = _bruteforce(0.98)

    # Between multiple optimal solutions, we select the most square one
    best = max(res, key=average_squaricity_criterion(text_polys))

    # Now join all polygons that are part of the same connected component
    groups = disjoint_groups([text_polys[b] for b in best], exact_disjoint_criterion)

    return [unary_union(list(g)) for g in groups], image_polys


def layout_detection(img):
    """Entry point for full layout detection."""

    # Binarize once in the beginning.
    binarized = otsu_binarization(img)

    # Dispatch to an impl function, as this function might be called recursively
    # with additional parameters.
    return impl_layout_detection(binarized)
