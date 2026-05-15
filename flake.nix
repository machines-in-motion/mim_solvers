{
  description = "Implementation of numerical solvers used in the Machines in Motion Laboratory";

  inputs.gepetto.url = "github:gepetto/nix";

  outputs =
    inputs:
    inputs.gepetto.lib.mkFlakoboros inputs (
      { lib, ... }:
      {
        # extraDevPyPackages = [ "mim-solvers" ];
        extraPyPackages = [ "pytest" ];
        extraPackages = [ "colcon" ];
        pyOverrideAttrs.mim-solvers = {
          patches = [ ];
          src = lib.fileset.toSource {
            root = ./.;
            fileset = lib.fileset.unions [
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
        };
      }
    );
}
