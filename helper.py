"""Helper functions."""
import numpy as np
from app.db.models import ImageClick


def get_coords_tuple(path):
    """Get co ordinates of clicks on an image as a tuple."""
    query = ImageClick.select().where(
        (ImageClick.path == path) &
        (ImageClick.x != "NULL") &
        (ImageClick.y != "NULL")
    )

    results_x = np.array([r.x for r in query]).astype(np.float)
    results_y = [-y for y in np.array([r.y for r in query]).astype(np.float)]

    coords = np.array([(r[0], r[1]) for r in zip(results_x, results_y)])
    return coords
