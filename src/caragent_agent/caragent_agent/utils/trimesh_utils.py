"""Depth-pyramid utilities are intentionally unavailable in CarAgent v1."""


def get_pyramids_mesh(*args, **kwargs):
    del args, kwargs
    raise NotImplementedError("CarAgent v1 does not have depth images for pyramid meshes.")


def check_pyramids_connection(*args, **kwargs):
    del args, kwargs
    return False, 0.0
