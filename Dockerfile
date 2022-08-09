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
        "libfindbin-libs-perl" \
        "less" && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install conda
# =============
FROM base as conda

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
FROM conda as builder

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
        "git" \
        "subversion" \
        "build-essential"  # need system `ar` for raremetal

RUN conda install --yes "boa" "conda-verify"

COPY recipes/raremetal raremetal
RUN conda mambabuild --no-anaconda-upload "raremetal"

COPY recipes/r-gmmat r-gmmat
RUN conda mambabuild --no-anaconda-upload "r-gmmat"

COPY recipes/r-saige r-saige
RUN conda mambabuild --no-anaconda-upload "r-saige"

# Install packages
# ================
FROM conda as install

COPY --from=builder /usr/local/miniconda/conda-bld /usr/local/miniconda/conda-bld
RUN conda install --yes --use-local \
        "plink" \
        "plink2" \
        "python >=3.10" \
        "raremetal" \
        "r-gmmat" \
        "r-saige" \
        "r-skat" && \
    sync && \
    rm -rf /usr/local/miniconda/conda-bld && \
    conda clean --yes --all --force-pkgs-dirs && \
    sync && \
    find /usr/local/miniconda/ -follow -type f -name "*.a" -delete && \
    sync

# Final
# =====
FROM base
COPY --from=install /usr/local/miniconda /usr/local/miniconda
