# GWAS Protocol (as a container)

## Usage

The container comes with all the software needed to generate summary statistics.

<ol>

<li>
<p>
You need to download the container file using one of the following commands. This will use approximately one gigabyte of storage.
</p>
<table>
<thead>
  <tr>
    <th><b>Container platform</b></th>
    <th><b>Version</b></th>
    <th><b>Command</b></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Singularity</td>
    <td>3.x</td>
    <td><code>wget <a href="http://download.gwas.science/singularity/gwas-protocol-latest.sif">http://download.gwas.science/singularity/ramp-latest.sif</code></a></td>
  </tr>
  <tr>
    <td>Docker</td>
    <td></td>
    <td><code>docker pull gwas.science/ramp:latest</code></td>
  </tr>
</tbody>
</table>
</li>

<li>
<p>
Next, start an interactive shell inside the container using one of the following commands.
</p>
<table>
<thead>
  <tr>
    <th><b>Container platform</b></th>
    <th><b>Command</b></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Singularity</td>
    <td><code>singularity shell --hostname localhost --bind ${working_directory}:/data --bind /tmp gwas-protocol-latest.sif</code></td>
  </tr>
  <tr>
    <td>Docker</td>
    <td>
        <code>docker run --interactive --tty --volume ${working_directory}:/data --bind /tmp gwas.science/gwas-protocol /bin/bash</code>
    </td>
  </tr>
</tbody>
</table>
</li>

## Development

To create a local development environment install [Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) and create a `.condarc` file in your home directory with the following contents:

```
channels:
  - conda-forge
  - bioconda
```

Next, create the environment using the following command:

```bash
micromamba create --name "gwas-protocol" \
  "conda-build" \
  "bcftools" "plink" "plink2" "tabix" "gcta" \
  "parallel" \
  "jupyterlab" "ipywidgets" \
  "python=3.12" "more-itertools" "psutil" "tqdm" "pyyaml" \
  "python-blosc2" "pyarrow" \
  "numpy" "scipy" "pandas" "threadpoolctl" "universal_pathlib" "tabulate" \
  "matplotlib" "seaborn" \
  "jax" "jaxlib=*=cpu*" "jaxtyping" "chex" "etils" "python-flatbuffers" \
  "mkl-include" "mkl" "c-blosc2" \
  "mypy" "pandas-stubs" "pyarrow-stubs" "types-psutil" "types-pyyaml" "types-seaborn" "types-setuptools" "types-tqdm" \
  "pytest-benchmark" "pytest-cov" \
  "cython" "gxx_linux-64>=13" "gcc_linux-64>=13" "sysroot_linux-64>=2.17" "zlib" "gdb"
```

Finally, install the `gwas` package using the following command:

```bash
pip install --no-deps --editable "src/gwas"
```

## Benchmark

```bash
data_path=/sc-projects/sc-proj-cc15-mb-enigma/genetics/development/opensnp
for sample_size in 100 500 3421; do
  mkdir -p "${sample_size}"
  pushd "${sample_size}" || exit 1
  benchmark --vcf $(for chromosome in $(seq 1 22); do echo ${data_path}/${sample_size}/chr${chromosome}.dose.vcf.zst; done) --output-directory . --method ramp --causal-variant-count 100 --simulation-count 1000 --seed 1000 --missing-value-pattern-count 10
  popd || exit 1
done

```
