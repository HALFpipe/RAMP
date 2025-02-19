def ceiling_division(numerator: int, denominator: int) -> int:
    quotient, remainder = divmod(numerator, denominator)
    return quotient + bool(remainder)


def round_up(number_to_round: int, multiple: int) -> int:
    """
    Rounds up a number to the nearest multiple.
    Adapted from https://stackoverflow.com/a/3407254

    Args:
        number_to_round (int): The number to be rounded up.
        multiple (int): The multiple to round up to.

    Returns:
        int: The rounded up number.

    """
    if multiple == 0:
        return number_to_round
    remainder = number_to_round % multiple
    if remainder == 0:
        return number_to_round
    return number_to_round + multiple - remainder
