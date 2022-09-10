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
        "gdb" \
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
        "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" \
        --output "miniconda.sh" &&  \
    bash miniconda.sh -b -p /usr/local/miniconda && \
    rm miniconda.sh && \
    conda install --channel "defaults" --yes "conda==4.13.0" && \
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

COPY recipes/qctool qctool
RUN conda mambabuild --no-anaconda-upload "qctool"

COPY recipes/bolt-lmm bolt-lmm
RUN conda mambabuild --no-anaconda-upload "bolt-lmm"

# Install packages
# ================
FROM conda as install

COPY --from=builder /usr/local/miniconda/conda-bld /usr/local/miniconda/conda-bld
RUN conda install --yes --use-local \
        "bcftools" \
        "bolt-lmm" \
        "gcta" \
        "parallel" \
        "plink" \
        "plink2" \
        "python >=3.10" \
        "qctool" \
        "raremetal" \
        "r-gmmat" \
        "r-saige" \
        "r-skat" \
        "tabix" && \
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
