{
  description = "CUDA 13.0 Dev Environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
  let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      config.allowUnfree = true;
    };
  in
  {
    devShells.${system}.default = pkgs.mkShell {
      buildInputs = [
        pkgs.cmake
        pkgs.ninja
        pkgs.clang-tools   # clangd
        pkgs.gcc14
        pkgs.gdb

        # CUDA for only for compilation
        pkgs.cudaPackages_13_0.cuda_nvcc
        pkgs.cudaPackages_13_0.cudatoolkit   # only headers
      ];

      shellHook = ''
        # nvcc compiler
        export CUDACXX=${pkgs.cudaPackages_13_0.cuda_nvcc}/bin/nvcc
        export CUDA_HOME=${pkgs.cudaPackages_13_0.cudatoolkit}

        # Use GCC 14
        export CC=${pkgs.gcc14}/bin/gcc
        export CXX=${pkgs.gcc14}/bin/g++

        # Unset LD_LIBRARY_PATH to pick up system driver
        unset LD_LIBRARY_PATH
      '';
    };
  };
}
