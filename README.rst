# gwas-protocol

# To create a local development environment run:
mamba create --name "gwas" \
  "mamba" \
  "python>=3.11" "more-itertools" \
  "jupyterlab" "ipywidgets" \
  "numpy" "scipy" "pandas" "pytorch<2" "networkx" \
  "matplotlib" "seaborn" \
  "bzip2" "p7zip>=15.09" \
  "c-blosc2" "msgpack-python" "ndindex" \
  "bcftools>=1.17" "plink" "plink2" "tabix" \
  "cython>=3b1" "mkl-include" "mypy" "pytest-benchmark" "threadpoolctl" \
  "gcc" "gxx" "sysroot_linux-64>=2.17"
