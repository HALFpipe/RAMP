# syntax=docker/dockerfile:1.4

FROM ubuntu:rolling as base

ENV LC_ALL="C.UTF-8" \
    LANG="C.UTF-8" \
    DEBIAN_FRONTEND="noninteractive" \
    DEBCONF_NOWARNINGS="yes" \
    PATH="/opt/conda/bin:$PATH"

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
    "ca-certificates" \
    "curl" \
    "gdb" \
    "git" "git-lfs" \
    "libc6-dbg" \
    "libencode-perl" \
    "libfindbin-libs-perl" \
    "less" \
    "valgrind" && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install conda
# =============
FROM base as conda

RUN curl --silent --show-error --location \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh" \
    --output "conda.sh" && \
    bash conda.sh -b -p /opt/conda && \
    rm conda.sh && \
    conda config --system --append channels "bioconda" && \
    conda config --system --prepend channels "conda-forge/label/cython_dev" && \
    sync && \
    mamba clean --yes --all --force-pkgs-dirs

# Build packages
# ==============
FROM conda as builder

RUN mamba install --yes "boa" "conda-verify"
COPY recipes/conda_build_config.yaml /root/conda_build_config.yaml

RUN --mount=source=recipes/bolt-lmm,target=/bolt-lmm \
    conda mambabuild --no-anaconda-upload "bolt-lmm" && \
    conda build purge

RUN --mount=source=recipes/dosage-convertor,target=/dosage-convertor \
    conda mambabuild --no-anaconda-upload "dosage-convertor" && \
    conda build purge

RUN --mount=source=recipes/gcta,target=/gcta \
    conda mambabuild --no-anaconda-upload "gcta" && \
    conda build purge

RUN --mount=source=recipes/qctool,target=/qctool \
    conda mambabuild --no-anaconda-upload "qctool" && \
    conda build purge

RUN --mount=source=recipes/raremetal,target=/raremetal \
    conda mambabuild --no-anaconda-upload "raremetal" && \
    conda build purge

RUN --mount=source=recipes/r-gmmat,target=/r-gmmat \
    conda mambabuild --no-anaconda-upload "r-gmmat" && \
    conda build purge

RUN --mount=source=recipes/r-saige,target=/r-saige \
    conda mambabuild --no-anaconda-upload "r-saige" && \
    conda build purge

RUN --mount=source=recipes/upload,target=/upload \
    conda mambabuild --no-anaconda-upload "upload" && \
    conda build purge

# mount .git folder too for setuptools_scm
RUN --mount=source=recipes/gwas,target=/gwas-protocol/recipes/gwas \
    --mount=source=src/gwas,target=/gwas-protocol/src/gwas \
    --mount=source=.git,target=/gwas-protocol/.git \
    cd gwas-protocol/recipes && \
    conda mambabuild --no-anaconda-upload --use-local "gwas" && \
    conda build purge

RUN conda index /opt/conda/conda-bld

# Install packages
# ================
FROM conda as install

COPY --from=builder /opt/conda/conda-bld /opt/conda/conda-bld
RUN mamba install --yes --use-local \
    "python=3.11" \
    "pytorch=*=cpu*" \
    "bcftools>=1.17" \
    "gemma" \
    "plink" \
    "plink2" \
    "r-skat" \
    "tabix" \
    "p7zip>=15.09" \
    "parallel" \
    "bolt-lmm" \
    "dosage-convertor" \
    "gcta" \
    "gwas" \
    "perl-vcftools-vcf" \
    "python-blosc2" \
    "qctool" \
    "raremetal" \
    "r-gmmat" \
    "r-saige" && \
    sync && \
    rm -rf /opt/conda/conda-bld && \
    mamba clean --yes --all --force-pkgs-dirs && \
    sync && \
    find /opt/conda/ -follow -type f -name "*.a" -delete && \
    sync

# Final
# =====
FROM base
COPY --from=install /opt/conda /opt/conda
