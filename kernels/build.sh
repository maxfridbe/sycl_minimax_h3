#!/usr/bin/env bash
# Build libh3sycl.so inside the h3-sycl-dev image (icpx + oneDNN). Run from the kernels/ directory, mounted at /k.
set -e
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
D=/opt/intel/oneapi/dnnl/2026.0
icpx -O3 -fsycl -fPIC -shared -std=c++20 -I$D/include h3sycl.cpp -L$D/lib -ldnnl -Wl,-rpath,'$ORIGIN' -o libh3sycl.so
cp -L $D/lib/libdnnl.so.3 . 2>/dev/null || true      # the PyTorch image has no shared oneDNN: ship ours beside the library
ls -la libh3sycl.so libdnnl.so.3
