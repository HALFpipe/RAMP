{
  description = "A basic flake with a shell";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
          config.allowUnfree = true;
        };
        compressionPackages = with pkgs; [ bcftools bzip2 htslib p7zip zstd ];

        python = (pkgs.python311.withPackages (py:
          with py; [
            blosc2
            cython_3
            matplotlib
            more-itertools
            msgpack
            mypy
            ndindex
            networkx
            numpy
            pandas
            pip
            psutil
            pytest
            pytest-benchmark
            pyyaml
            scipy
            seaborn
            setuptools-rust
            setuptools-scm
            threadpoolctl
            torch-bin
            tqdm
            types-pyyaml
            zstandard
          ]));
        rust = pkgs.rust-bin.beta.latest.default.override {
          extensions = [ "rust-src" ];
        };
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs;
            [ mkl plink-ng python rust rust-analyzer-unwrapped ]
            ++ compressionPackages;
          RUST_SRC_PATH = "${rust}/lib/rustlib/src/rust/library";
        };
      });
}
