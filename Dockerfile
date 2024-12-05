FROM ubuntu:rolling AS base

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
FROM base AS conda

ENV PATH="/opt/conda/bin:$PATH"
RUN curl --silent --show-error --location \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" \
    --output "conda.sh" && \
    bash "conda.sh" -b -p "/opt/conda" && \
    rm "conda.sh" && \
    conda config --system  \
    --set "solver" "libmamba" \
    --set "conda_build.pkg_format" "2" && \
    conda config --system --append "channels" "bioconda" && \
    sync && \
    conda clean --yes --all --force-pkgs-dirs

# Build packages
# ==============
FROM conda AS builder
RUN conda install --yes "c-compiler" "conda-build"
COPY recipes/conda_build_config.yaml /root/conda_build_config.yaml

RUN cat <<EOF > "/usr/bin/retry"
#!/bin/bash
set -euo pipefail
timeout="1"
attempt="1"
until "\$@"; do
    if [ "\${attempt}" -ge "5" ]; then
        exit "$?"
    fi
    sleep "\${timeout}"
    timeout=\$((timeout * 10))
    attempt=\$((attempt + 1))
done
EOF
RUN chmod "+x" "/usr/bin/retry"

# https://danieldk.eu/Posts/2020-08-31-MKL-Zen.html
RUN echo "int mkl_serv_intel_cpu_true() {return 1;}" | \
    gcc -x "c" -shared -fPIC -o "/opt/libfakeintel.so" -

FROM builder AS dosage-convertor
RUN --mount=source=recipes/dosage-convertor,target=/dosage-convertor \
    --mount=type=cache,target=/opt/conda/pkgs \
    retry conda build --no-anaconda-upload --numpy "2.0" "dosage-convertor"

FROM builder AS metal
RUN --mount=source=recipes/metal,target=/metal \
    --mount=type=cache,target=/opt/conda/pkgs \
    retry conda build --no-anaconda-upload --numpy "2.0" --use-local "metal"

FROM builder AS pynose
RUN --mount=source=recipes/pynose,target=/pynose \
    --mount=type=cache,target=/opt/conda/pkgs \
    retry conda build --no-anaconda-upload --numpy "2.0" --use-local "pynose"

FROM builder AS qctool
RUN --mount=source=recipes/qctool,target=/qctool \
    --mount=type=cache,target=/opt/conda/pkgs \
    retry conda build --no-anaconda-upload --numpy "2.0" --use-local "qctool"

FROM builder AS raremetal
RUN --mount=source=recipes/raremetal,target=/raremetal \
    --mount=type=cache,target=/opt/conda/pkgs \
    retry conda build --no-anaconda-upload --numpy "2.0" --use-local "raremetal"

FROM builder AS raremetal-debug
RUN --mount=source=recipes/raremetal,target=/raremetal \
    --mount=source=recipes/raremetal-debug,target=/raremetal-debug \
    --mount=type=cache,target=/opt/conda/pkgs \
    retry conda build --no-anaconda-upload --numpy "2.0" --use-local "raremetal-debug"

FROM builder AS r-genomicsem
RUN --mount=source=recipes/r-genomicsem,target=/r-genomicsem \
    --mount=type=cache,target=/opt/conda/pkgs \
    retry conda build --no-anaconda-upload --numpy "2.0" --use-local "r-genomicsem"

FROM builder AS r-gmmat
RUN --mount=source=recipes/r-gmmat,target=/r-gmmat \
    --mount=type=cache,target=/opt/conda/pkgs \
    retry conda build --no-anaconda-upload --numpy "2.0" --use-local "r-gmmat"

FROM builder AS upload
RUN --mount=source=recipes/upload,target=/upload \
    --mount=type=cache,target=/opt/conda/pkgs \
    retry conda build --no-anaconda-upload --numpy "2.0" --use-local "upload"

FROM builder AS ldsc
COPY --from=pynose /opt/conda/conda-bld /opt/conda/conda-bld
RUN --mount=source=recipes/ldsc,target=/ldsc \
    --mount=type=cache,target=/opt/conda/pkgs \
    retry conda build --no-anaconda-upload --numpy "2.0" --use-local "ldsc"

FROM builder AS gwas
COPY --from=dosage-convertor /opt/conda/conda-bld /opt/conda/conda-bld
COPY --from=ldsc /opt/conda/conda-bld /opt/conda/conda-bld
COPY --from=metal /opt/conda/conda-bld /opt/conda/conda-bld
COPY --from=qctool /opt/conda/conda-bld /opt/conda/conda-bld
COPY --from=raremetal /opt/conda/conda-bld /opt/conda/conda-bld
COPY --from=raremetal-debug /opt/conda/conda-bld /opt/conda/conda-bld
COPY --from=r-genomicsem /opt/conda/conda-bld /opt/conda/conda-bld
COPY --from=r-gmmat /opt/conda/conda-bld /opt/conda/conda-bld
COPY --from=upload /opt/conda/conda-bld /opt/conda/conda-bld
RUN conda index /opt/conda/conda-bld
# Mount .git folder too for setuptools_scm
RUN --mount=source=recipes/gwas,target=/gwas-protocol/recipes/gwas \
    --mount=source=src/gwas,target=/gwas-protocol/src/gwas \
    --mount=source=.git,target=/gwas-protocol/.git \
    --mount=type=cache,target=/opt/conda/pkgs \
    retry conda build --no-anaconda-upload --numpy "2.0" --use-local "gwas-protocol/recipes/gwas"

# Install packages
# ================
FROM conda AS install

COPY --from=gwas /opt/conda/conda-bld /opt/conda/conda-bld
RUN --mount=type=cache,target=/opt/conda/pkgs \
    conda install --yes --channel "/opt/conda/conda-bld" \
    "parallel" \
    "dosage-convertor" \
    "qctool" \
    "raremetal-debug" \
    "python=3.12" \
    "jaxlib=*=cpu*" \
    "gwas" && \
    conda create --yes --name "bgenix" "bgenix" && \
    conda create --yes --name "regenie" "regenie" && \
    conda create --yes --channel "/opt/conda/conda-bld" --name "r-genomicsem" "r-genomicsem" "r-qqman" && \
    conda create --yes --name "r-saige" "r-saige" && \
    sync && \
    rm -rf /opt/conda/conda-bld && \
    conda clean --yes --all --force-pkgs-dirs

# Final
# =====
FROM base AS final
COPY --from=install --chown=ubuntu:ubuntu /opt/conda /opt/conda

# Ensure that we can link to libraries installed via conda
ENV PATH="/opt/conda/bin:$PATH" \
    CPATH="/opt/conda/include:${CPATH}"
RUN echo /opt/conda/lib > /etc/ld.so.conf.d/conda.conf && \
    ldconfig

ENV LD_PRELOAD="/opt/libfakeintel.so"
COPY --from=builder --chown=ubuntu:ubuntu /opt/libfakeintel.so /opt/libfakeintel.so

FROM final AS test

RUN conda install --yes \
    "mypy" "pytest-benchmark" \
    "gcc>=13.1" "gxx>=13.1" "binutils" \
    "mkl-include" "zlib" \
    "sysroot_linux-64>=2.17" && \
    sync && \
    rm -rf /opt/conda/conda-bld && \
    conda clean --yes --all --force-pkgs-dirs

FROM final
