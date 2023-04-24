# -*- coding: utf-8 -*-
import logging

logger = logging.getLogger("gwas")


def setup_logging(level: str | int) -> None:
    logging.basicConfig(
        level=level,
        format=(
            "%(asctime)s [%(levelname)8s] %(funcName)s: "
            "%(message)s (%(filename)s:%(lineno)s)"
        ),
    )
