def parse_frame_dimensions(dimensions: str) -> dict:
    """
    Format: lens_width-bridge-temple-lens_height
    Example: 51-18-142-41
    """
    lens_width, bridge, temple, lens_height = map(int, dimensions.split("-"))

    return {
        "lens_width": lens_width,
        "bridge": bridge,
        "temple": temple,
        "lens_height": lens_height
    }


def compute_fitting_height(lens_height: float) -> float:
    """
    Fitting height = distance from pupil center to bottom of lens.
    Client-approved heuristic: 2/3 of lens height.
    """
    return round(lens_height * (2 / 3), 2)

