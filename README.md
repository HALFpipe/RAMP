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
    <td><code>wget <a href="http://download.gwas.science/singularity/gwas-protocol-latest.sif">http://download.gwas.science/singularity/gwas-protocol-latest.sif</code></a></td>
  </tr>
  <tr>
    <td>Docker</td>
    <td></td>
    <td><code>docker pull gwas.science/gwas-protocol:latest</code></td>
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

To create a local development environment install [Miniforge](https://github.com/conda-forge/miniforge) and create a `.condarc` file in your home directory with the following contents:

```
channels:
  - conda-forge
  - bioconda
```

Then update your `.bashrc` or `.zshrc` with `mamba init`. This will allow you to use the `conda` command.

Next, install `mamba` using `conda install mamba` and then create the environment using the following command:

```bash
conda create --name "gwas-protocol" \
  "conda-build" \
  "bzip2" "p7zip" \
  "bcftools" "plink" "plink2" "tabix" \
  "gcta" \
  "parallel" \
  "jupyterlab" "ipywidgets" \
  "python=3.12" "more-itertools" \
  "numpy" "scipy" "pandas" "threadpoolctl" \
  "seaborn" \
  "jax" "jaxlib=*=cpu*" "jaxtyping" "chex" \
  "mkl-include" "mkl" \
  "mypy" "types-pyyaml"  \
  "pytest-benchmark" "pytest-cov" \
  "setuptools-rust" "cython>=3" "zlib" \
  "gxx_linux-64>=13" "gcc_linux-64>=13" "rust" "sysroot_linux-64>=2.17"
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
