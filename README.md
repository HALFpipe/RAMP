GWAS Protocol (as a container)
==============================

Usage
-----

The container comes with all the software needed to genrate summary statistics.
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

Development
-----------

To create a local development environment run:

```bash
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
```