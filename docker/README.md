<!--
SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file

SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
-->

# Dockerfiles

Those Dockerfiles are mainly used for CI.
Each backend/compiler has a corresponding Dockerfile.

# Contributing

## Update the Dockerfiles

If you have to update a Dockerfile, please do it in a specific PR.
This is cumbersome, but it guarantees that Docker images are not messed up in the registry.

By instance, let's say you want to add a new backend, which requires a new Dockerfile:

- Step 1: Create a branch for the new Dockerfile. Have it reviewed and merged. The new image will be added to the registry when the branch is merged.
- Step 2: Create a branch for the new backend, which CI uses the newly added Dockerfile.

## CMake installation in Dockerfiles

As the project requires CMake v3.23 at least, and as some Dockerfiles are based on Ubuntu 20.04 images, CMake has to be installed manually.
The installer is downloaded, its signature is verified, then its checksum is verified.
To check the signature, the public key of the person who signed the binary is required.
This public key can be extracted from the key ID.

When updating the Dockerfiles for a newer version of CMake (if needed), the process to get the right public key is as follows:

1. Identify the release on GitHub (e.g. https://github.com/Kitware/CMake/releases/tag/v3.23.2);
2. Copy the key ID in the line "PGP sign by XXXXXXXX";
3. Paste it in `https://keys.openpgp.org/` to retrieve the URL of the public key file;
4. Copy the last part in the URL `https://keys.openpgp.org/vks/v1/by-fingerprint/YYYYYYYY`;
5. Update the Dockerfiles with this value.
