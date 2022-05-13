FROM ubuntu as base

ENV LC_ALL="C.UTF-8" \
    LANG="C.UTF-8" \
    DEBIAN_FRONTEND="noninteractive" \
    DEBCONF_NOWARNINGS="yes" \
    PATH="/usr/local/miniconda/bin:$PATH" \
    CPATH="/usr/local/miniconda/include:$CPATH" \
    LD_LIBRARY_PATH="/usr/local/miniconda/lib:$LD_LIBRARY_PATH"

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
        "ca-certificates" \
        "curl" \
        "libencode-perl" \
        "libfindbin-libs-perl" && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Build packages
# ==============
FROM continuumio/miniconda3 as conda-builder

RUN conda update --channel "defaults" --yes "conda" && \
    conda config --system --add channels "bioconda" && \
    conda config --system --add channels "conda-forge" && \
    conda config --set channel_priority "strict" && \
    conda install --yes "boa" "conda-verify"

COPY recipes .

RUN conda mambabuild --no-anaconda-upload "r-epacts"
RUN conda mambabuild --no-anaconda-upload "r-gmmat"
RUN conda mambabuild --no-anaconda-upload "r-saige"

RUN conda mambabuild --no-anaconda-upload "r-mathlib"
RUN conda mambabuild --no-anaconda-upload "raremetal"

# Install packages
# ================
FROM base

COPY --from=conda-builder /opt/conda/conda-bld /usr/local/miniconda/conda-bld
RUN curl --silent --show-error --location \
        "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" \
        --output "miniconda.sh" &&  \
    bash miniconda.sh -b -p /usr/local/miniconda && \
    rm miniconda.sh && \
    conda update --channel "defaults" --yes "conda" && \
    conda config --system --add channels "bioconda" && \
    conda config --system --add channels "conda-forge" && \
    conda config --set channel_priority "strict" && \
    conda install --yes \
        "python=3.10" && \
    conda install --yes \
        "raremetal" \
        "r-epacts" \
        "r-gmmat" \
        "r-saige" \
        "bioconductor-snprelate" && \
    sync && \
    conda build purge-all && \
    conda clean --yes --all --force-pkgs-dirs && \
    sync && \
    find /usr/local/miniconda/ -follow -type f -name "*.a" -delete && \
    sync
