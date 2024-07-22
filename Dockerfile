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
    bash "conda.sh" -b -p "/opt/conda" && \
    rm "conda.sh" && \
    conda config --system  --set "solver" "libmamba" && \
    conda config --system --append "channels" "bioconda" && \
    sync && \
    conda clean --yes --all --force-pkgs-dirs

# Build packages
# ==============
FROM conda as builder
RUN conda install --yes "c-compiler" "conda-build"
COPY recipes/conda_build_config.yaml /root/conda_build_config.yaml

# https://danieldk.eu/Posts/2020-08-31-MKL-Zen.html
RUN echo "int mkl_serv_intel_cpu_true() {return 1;}" | \
    gcc -x "c" -shared -fPIC -o "/opt/libfakeintel.so" -

RUN --mount=source=recipes/dosage-convertor,target=/dosage-convertor \
    conda build --no-anaconda-upload --numpy "1.26" "dosage-convertor"
RUN --mount=source=recipes/qctool,target=/qctool \
    conda build --no-anaconda-upload --numpy "1.26" --use-local "qctool"
RUN --mount=source=recipes/raremetal,target=/raremetal \
    --mount=source=recipes/raremetal-debug,target=/raremetal-debug \
    conda build --no-anaconda-upload --numpy "1.26" --use-local "raremetal" && \
    conda build --no-anaconda-upload --numpy "1.26" --use-local "raremetal-debug"
RUN --mount=source=recipes/r-gmmat,target=/r-gmmat \
    conda build --no-anaconda-upload --numpy "1.26" --use-local "r-gmmat"
# RUN --mount=source=recipes/upload,target=/upload \
#     conda build --no-anaconda-upload --numpy "1.26" --use-local "upload"
# Mount .git folder too for setuptools_scm
RUN --mount=source=recipes/gwas,target=/gwas-protocol/recipes/gwas \
    --mount=source=src/gwas,target=/gwas-protocol/src/gwas \
    --mount=source=.git,target=/gwas-protocol/.git \
    conda build --no-anaconda-upload --numpy "1.26" --use-local "gwas-protocol/recipes/gwas"
RUN conda build purge

RUN conda index /opt/conda/conda-bld

# Install packages
# ================
FROM conda as install

COPY --from=builder /opt/conda/conda-bld /opt/conda/conda-bld
RUN conda install --yes --use-local \
    "parallel" \
    "dosage-convertor" \
    "qctool" \
    "raremetal-debug" \
    "python=3.12" \
    "jaxlib=*=cpu*" \
    "gwas" && \
    conda create --name "bgenix" "bgenix" && \
    conda create --name "regenie" "regenie" && \
    conda create --name "r-saige" "r-saige" && \
    sync && \
    rm -rf /opt/conda/conda-bld && \
    conda clean --yes --all --force-pkgs-dirs

# Final
# =====
FROM base
COPY --from=install --chown=ubuntu:ubuntu /opt/conda /opt/conda

# Ensure that we can link to libraries installed via conda
ENV PATH="/opt/conda/bin:$PATH" \
    CPATH="/opt/conda/include:${CPATH}"
RUN echo /opt/conda/lib > /etc/ld.so.conf.d/conda.conf && \
    ldconfig

ENV LD_PRELOAD="/opt/libfakeintel.so"
COPY --from=builder --chown=ubuntu:ubuntu /opt/libfakeintel.so /opt/libfakeintel.so
