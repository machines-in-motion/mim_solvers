{
  description = "Implementation of numerical solvers used in the Machines in Motion Laboratory";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    crocoddyl = {
      # Devel branch because boost::shared_ptr -> std::shared_ptr
      url = "github:loco-3d/crocoddyl/master";
      inputs.flake-parts.follows = "flake-parts";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      systems = inputs.nixpkgs.lib.systems.flakeExposed;
      perSystem =
        { pkgs, self', ... }:
        let
          # Define pythonSupport based on some condition, e.g., based on system or user preference
          pythonSupport = true;
        in
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
            mim-solvers = pkgs.python3Packages.mim-solvers.overrideAttrs (old: {
              # Remove the dependency to python3Packages.crocoddyl and the old crocoddyl package
              propagatedBuildInputs = 
                [ inputs.crocoddyl.packages.${pkgs.system}.default ]
                ++ pkgs.lib.optionals pythonSupport [
                  pkgs.python3Packages.osqp
                  pkgs.python3Packages.proxsuite
                  pkgs.python3Packages.scipy
                ]
                ++ pkgs.lib.optionals (!pythonSupport) [
                  pkgs.proxsuite
                ];

              # Keep the src for the mim-solvers package itself
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
