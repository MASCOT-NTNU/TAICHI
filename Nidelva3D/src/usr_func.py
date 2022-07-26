""" This object contains useful user-defined functions that are applied throughout the whole package.

"""


def is_even(value: int) -> bool:
    """
    Checks if a given integer even or not.

    Args:
        value: integer value

    Returns: True if it is even, False or else.

    """
    if value % 2 == 0:
        return True
    else:
        return False

