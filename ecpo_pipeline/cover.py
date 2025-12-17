import networkx as nx
import numpy as np

import _cover_heuristic

from PIL import Image
from shapely.ops import unary_union, polygonize
from skimage.filters import threshold_otsu


def otsu_binarization(img):
    """Apply binarization to an image.

    Not refined enough for OCR, but well worth it for layout detection.
    """
    img = np.array(Image.fromarray(img).convert("L"))
    thr = threshold_otsu(img)
    return (img > thr) * 255


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
    from ecpo_pipeline.detect import crop_polygon

    cropped_img, mask_cropped = crop_polygon(binary, poly)
    return np.sum(mask_cropped & (cropped_img == 0))


def squaricity(poly):
    """A measure for how square-like a polygon is.

    Values are always in [0, 1] with 1 is an acutal square.
    """
    return 16 * poly.area / (poly.length * poly.length)


def average_squaricity_criterion(polys):
    """Criterion for average squaricity for a set of polygons."""

    def _average_squaricity_criterion(indices):
        return sum((squaricity(polys[i]) for i in indices), 0.0) / len(indices)

    return _average_squaricity_criterion


def overlap(p, q):
    """Overlap percentage of two polygons.

    Values are always in [0, 1] with 1 for identical polygons.
    """
    return p.intersection(q).area / p.union(q).area


def exact_disjoint_criterion(p, q):
    return p.intersection(q).area == 0


def filter_redundant_polys(polys):
    """Filter polygons that are almost identical with others."""

    uf = nx.utils.UnionFind(polys)

    for i, p in enumerate(polys):
        for j, q in enumerate(polys):
            if i < j:
                if overlap(p, q) > 0.9:
                    uf.union(p, q)

    return [unary_union(list(s)) for s in uf.to_sets()]


def intersection_edges(polys):
    """Calculate the intersection graph for a set of polygons."""

    from ecpo_pipeline.detect import disjoint_criterion

    edges = []
    for i, p in enumerate(polys):
        for j, q in enumerate(polys):
            if i != j:
                if not exact_disjoint_criterion(p, q):
                    # if not disjoint_criterion(p, q):
                    edges.append((i, j))

    return edges


def maximum_cover_patch_heuristic(img, polys):
    """The heuristic that maximizes patch number while covering ."""

    img = otsu_binarization(img)
    polys = filter_redundant_polys(polys)

    atomics, poly_atomics = calculate_atomics(polys)
    atomics_values = [black_content(img, a) for a in atomics]
    edges = intersection_edges(polys)

    res = _cover_heuristic.find_optimal_cover(0.98, edges, poly_atomics, atomics_values)
    best = max(res, key=average_squaricity_criterion(polys))

    from ecpo_pipeline.detect import disjoint_groups

    groups = disjoint_groups([polys[b] for b in best], exact_disjoint_criterion)

    return [unary_union(list(g)) for g in groups]
