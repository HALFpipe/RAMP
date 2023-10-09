{
  description = "A basic flake with a shell";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
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

        blosc2 =
          pkgs.python311Packages.blosc2.overridePythonAttrs (old: rec {
            pname = "blosc2";
            version = "2.2.7";
            src = pkgs.fetchPypi {
              inherit version pname;
              hash = "sha256-e22AVEbFYLJgA9H5B+e9No7dgtIygz6fO2F+LIgQjV4=";
            };
          });
        setuptools-rust =
          pkgs.python311Packages.setuptools-rust.overridePythonAttrs (old: rec {
            pname = "setuptools-rust";
            version = "1.7.0";
            format = "pyproject";
            src = pkgs.fetchPypi {
              inherit version pname;
              hash = "sha256-xxAJmZSCNaOK5+VV/hmapmwlPcOEsSX12FRzv4Hq46M=";
            };
          });
        pythonPackages = with pkgs.python311Packages; [
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
          torch
          tqdm
          types-pyyaml
          zstandard
        ];
        rust = pkgs.rust-bin.beta.latest.default.override {
          extensions = [ "rust-src" ];
        };
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs;
            [ mkl plink-ng python311 rust rust-analyzer-unwrapped ]
            ++ compressionPackages ++ pythonPackages;
          RUST_SRC_PATH = "${rust}/lib/rustlib/src/rust/library";
        };
      });
}
