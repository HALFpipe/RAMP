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
      lib = pkgs.lib;

      # Make scripts from the Python project available
      attrs = builtins.fromTOML (builtins.readFile ./src/gwas/pyproject.toml);
      write = name: value:
        let
          match = lib.splitString ":" value;
          package = builtins.elemAt match 0;
          function = builtins.elemAt match 1;
        in
        pkgs.writers.makeScriptWriter { interpreter = "${pkgs.coreutils}/bin/env python"; } "/bin/${name}" ''
          import sys
          from ${package} import ${function}
          if __name__ == "__main__":
            sys.exit(${function}())
        '';
      scripts = lib.mapAttrsToList write attrs.project.scripts;
    in
    {
      devShells.x86_64-linux.default = pkgs.mkShell {
        packages = [ pkgs.clang pkgs.clang-tools pkgs.gcc pkgs.gdb pkgs.micromamba pkgs.ruff ] ++ scripts;
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
