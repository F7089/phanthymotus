#!/bin/bash
# Do not compile this on the Jetson host or inside a running eval container.
# libort_cuda_mem_hook.so is built in perception/Dockerfile.jetson against
# that image's glibc. Host gcc on a newer userspace produced GLIBC_2.34
# and JP5 containers (glibc 2.31) failed to start python.
echo "compile on image build only: see perception/Dockerfile.jetson" >&2
exit 1
