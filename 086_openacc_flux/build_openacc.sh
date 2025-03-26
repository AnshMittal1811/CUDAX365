#!/usr/bin/env bash
set -euo pipefail
nvc++ -acc -Minfo=accel openacc_flux.c -o openacc_flux
./openacc_flux
