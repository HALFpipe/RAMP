FROM ubuntu:rolling as base

ENV LC_ALL="C.UTF-8" \
    LANG="C.UTF-8" \
    DEBIAN_FRONTEND="noninteractive" \
    DEBCONF_NOWARNINGS="yes"

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

ENV PATH="/opt/conda/bin:$PATH"
RUN curl --silent --show-error --location \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" \
    --output "conda.sh" && \
    bash "conda.sh" -b -p /opt/conda && \
    rm "conda.sh" && \
    conda config --system  --set "solver" "libmamba" && \
    conda config --system --append "channels" "bioconda" && \
    sync && \
    conda clean --yes --all --force-pkgs-dirs

# Build packages
# ==============
FROM conda as builder
RUN conda install --yes "conda-build"
COPY recipes/conda_build_config.yaml /root/conda_build_config.yaml

RUN --mount=source=recipes/dosage-convertor,target=/dosage-convertor \
    conda build --no-anaconda-upload --numpy "1.26" "dosage-convertor" && \
    conda build purge
RUN --mount=source=recipes/qctool,target=/qctool \
    conda build --no-anaconda-upload --numpy "1.26" --use-local "qctool" && \
    conda build purge
RUN --mount=source=recipes/raremetal,target=/raremetal \
    conda build --no-anaconda-upload --numpy "1.26" --use-local "raremetal" && \
    conda build purge
RUN --mount=source=recipes/r-gmmat,target=/r-gmmat \
    conda build --no-anaconda-upload --numpy "1.26" --use-local "r-gmmat" && \
    conda build purge
RUN --mount=source=recipes/r-saige,target=/r-saige \
    conda build --no-anaconda-upload --numpy "1.26" --use-local "r-saige" && \
    conda build purge
RUN --mount=source=recipes/upload,target=/upload \
    conda build --no-anaconda-upload --numpy "1.26" --use-local "upload" && \
    conda build purge
# Mount .git folder too for setuptools_scm
RUN --mount=source=recipes/gwas,target=/gwas-protocol/recipes/gwas \
    --mount=source=src/gwas,target=/gwas-protocol/src/gwas \
    --mount=source=.git,target=/gwas-protocol/.git \
    conda build --no-anaconda-upload --numpy "1.26" --use-local "gwas-protocol/recipes/gwas" && \
    conda build purge

RUN conda index /opt/conda/conda-bld

# Install packages
# ================
FROM conda as install

COPY --from=builder /opt/conda/conda-bld /opt/conda/conda-bld
RUN mamba install --yes --use-local \
    "python=3.12" \
    "jaxlib=*=cpu*" \
    "bcftools>=1.17" \
    "gemma" \
    "plink" \
    "plink2" \
    "r-skat" \
    "tabix" \
    "p7zip>=15.09" \
    "parallel" \
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
    mamba clean --yes --all --force-pkgs-dirs

# Final
# =====
FROM base
COPY --from=install /opt/conda /opt/conda

# Ensure that we can link to libraries installed via conda
ENV PATH="/opt/conda/bin:$PATH" \
    CPATH="/opt/conda/include:${CPATH}"
RUN echo /opt/conda/lib > /etc/ld.so.conf.d/conda.conf && \
    ldconfig
