# -*- coding: utf-8 -*-
def chromosome_to_int(chromosome: int | str) -> int:
    if chromosome == "X":
        return 23
    elif isinstance(chromosome, int):
        return chromosome
    raise ValueError(f"Unknown chromsome \"{chromosome}\"")
