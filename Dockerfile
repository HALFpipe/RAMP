FROM ubuntu as base

ENV LC_ALL="C.UTF-8" \
    LANG="C.UTF-8" \
    DEBIAN_FRONTEND="noninteractive" \
    DEBCONF_NOWARNINGS="yes" \
    PATH="/usr/local/miniconda/bin:$PATH"

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
        "ca-certificates" \
        "curl" \
        "libencode-perl" \
        "libfindbin-libs-perl" && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN curl --silent --show-error --location \
        "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" \
        --output "miniconda.sh" &&  \
    bash miniconda.sh -b -p /usr/local/miniconda && \
    rm miniconda.sh && \
    conda update --channel "defaults" --yes "conda" && \
    conda config --system --add channels "bioconda" && \
    conda config --system --add channels "conda-forge" && \
    conda config --set channel_priority "strict" && \
    sync && \
    conda clean --yes --all --force-pkgs-dirs

# Build packages
# ==============
FROM base as builder

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
        "git" \
        "subversion" \
        "build-essential"  # need system `ar` for raremetal

RUN conda install --yes "boa" "conda-verify"

COPY recipes/raremetal raremetal
RUN conda mambabuild --no-anaconda-upload "raremetal"

COPY recipes/r-epacts r-epacts
RUN conda mambabuild --no-anaconda-upload "r-epacts"

COPY recipes/r-gmmat r-gmmat
RUN conda mambabuild --no-anaconda-upload "r-gmmat"

COPY recipes/r-saige r-saige
RUN conda mambabuild --no-anaconda-upload "r-saige"

# Install packages
# ================
FROM base

COPY --from=builder /usr/local/miniconda/conda-bld /usr/local/miniconda/conda-bld
RUN conda install --yes \
        "bioconductor-snprelate" \
        "python >=3.10" \
        "raremetal" \
        "r-epacts" \
        "r-gmmat" \
        "r-saige" && \
    sync && \
    conda build purge-all && \
    conda clean --yes --all --force-pkgs-dirs && \
    sync && \
    find /usr/local/miniconda/ -follow -type f -name "*.a" -delete && \
    sync
