{
  description = "A basic flake with a shell";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/release-24.05";
  outputs = { self, nixpkgs }:
    let
      pkgs = import nixpkgs {
        system = "x86_64-linux";
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      };
    in
    {
      devShells.x86_64-linux.default = pkgs.mkShell {
        packages = [ pkgs.clang pkgs.clang-tools pkgs.gcc pkgs.gdb pkgs.micromamba pkgs.ruff ];
        shellHook = ''
          export PYTHONBREAKPOINT=ipdb.set_trace
          export PYTHONDONTWRITEBYTECODE=1
          export PYTHONUNBUFFERED=1
          export PYTHONPATH="$(git rev-parse --show-toplevel)/src/gwas/src"
          eval "$(micromamba shell hook --shell=posix)"
          micromamba activate gwas-protocol
        '';
      };
    };
}
