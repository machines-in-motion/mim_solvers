{
  description = "Implementation of numerical solvers used in the Machines in Motion Laboratory";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    crocoddyl = {
      # Devel branch because boost::shared_ptr -> std::shared_ptr
      url = "github:loco-3d/crocoddyl/devel";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      systems = inputs.nixpkgs.lib.systems.flakeExposed;
      perSystem =
        { pkgs, self', ... }:
        {
          apps.default = {
            type = "app";
            program = pkgs.python3.withPackages (_: [ self'.packages.default ]);
          };
          devShells.default = pkgs.mkShell {
            inputsFrom = [ self'.packages.default ];
            packages = [ (pkgs.python3.withPackages (p: [p.tomlkit])) ]; # for "make release"
          };
          packages = {
            default = self'.packages.mim-solvers;
            mim-solvers = pkgs.python3Packages.mim-solvers.overrideAttrs (_: {
              src = pkgs.lib.fileset.toSource {
                root = ./.;
                fileset = pkgs.lib.fileset.unions [
                  ./benchmarks
                  ./bindings
                  ./examples
                  ./include
                  ./python
                  ./src
                  ./tests
                  ./CMakeLists.txt
                  ./package.xml
                ];
              };
            });
          };
        };
    };
}
