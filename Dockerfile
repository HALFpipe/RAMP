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
        "libz-dev" \
        "libzstd-dev" && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Compile RAREMETAL
# =================
FROM base as raremetal

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
        "build-essential" \
        "cmake" \
        "gfortran" \
        "git" \
        "libcurl4-openssl-dev" \
        "liblzma-dev" \
        "python3-pip" \
        "r-mathlib"

RUN pip install --no-cache-dir "cget"

RUN git clone --depth 1 https://github.com/statgen/raremetal && \
    cd raremetal && \
    cget ignore xz zlib zstd && \
    cget install --file requirements.txt && \
    mkdir build && \
    cd build && \
    cmake \
        -DCMAKE_BUILD_TYPE="Release" \
        -DCMAKE_TOOLCHAIN_FILE="../cget/cget/cget.cmake" \
        -DBUILD_TESTS="1" \
        .. && \
    make && \
    make test || true && \
    cd ../..

# Build Conda packages
# ====================
FROM continuumio/miniconda3 as conda-builder

RUN conda update --channel "defaults" --yes "conda" && \
    conda config --system --add channels "bioconda" && \
    conda config --system --add channels "conda-forge" && \
    conda config --set channel_priority "strict" && \
    conda install --yes "boa" "conda-verify"

COPY r-saige r-saige
RUN conda mambabuild --no-anaconda-upload r-saige

COPY r-gmmat r-gmmat
RUN conda mambabuild --no-anaconda-upload r-gmmat

COPY r-epacts r-epacts
RUN conda mambabuild --no-anaconda-upload r-epacts

# Install Conda
# =============
FROM base as conda

COPY --from=conda-builder /opt/conda/conda-bld conda-bld
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
    conda install --yes --channel "file:///conda-bld" \
        "r-saige" \
        "r-gmmat" \
        "r-epacts" && \
    sync && \
    conda clean --yes --all --force-pkgs-dirs && \
    sync && \
    find /usr/local/miniconda/ -follow -type f -name "*.a" -delete && \
    sync

# Create combined container
# =========================
FROM base

COPY --from=raremetal /raremetal/build/raremetal /raremetal/build/raremetalworker /usr/local/bin/
COPY --from=conda /usr/local/miniconda /usr/local/miniconda
