# -*- coding: utf-8 -*-
from pathlib import Path

import cyvcf2
import numpy as np
import pandas as pd

from gwas.vcf.base import AbstractVCFFile


class CyVCFReader(AbstractVCFFile):
    def __init__(self, file_path: Path | str) -> None:
        super().__init__(file_path)
        self.vcf: cyvcf2.VCF = cyvcf2.VCF(file_path)
        self.samples = list(self.vcf.samples)
        self.variants = []

    def _read_header(self):
        return {h for h in self.vcf.header_iter()}

    def set_samples(self, samples: set[str]):
        self.samples = [sample for sample in self.samples if sample in samples]

    def set_variants_from_cutoffs(
        self,
        minor_allele_frequency_cutoff: float = -np.inf,
        r_squared_cutoff: float = -np.inf,
    ):
        filtered_variants = []
        for v in self.vcf:
            m_af = float(v.INFO.get("MAF"))
            r_sq = float(v.INFO.get("R2"))
            if m_af > minor_allele_frequency_cutoff and r_sq > r_squared_cutoff:
                filtered_variants.append(v)
        self.variants = filtered_variants

    def make_data_frame(self, vcf_variants: list[cyvcf2.Variant]) -> pd.DataFrame:
        data_frame = [
            {
                "chromosome_int": v.CHROM,
                "position": v.POS,
                "reference_allele": v.REF,
                "alternate_allele": v.ALT,
                # 'is_imputed': v.?,
                #'alternate_allele_frequency': v.?
                "minor_allele_frequency": v.INFO.get("MAF"),
                "r_squared": v.INFO.get("R2"),
                "format_str": v.FORMAT,
            }
            for v in vcf_variants
        ]
        return pd.DataFrame(data_frame)
