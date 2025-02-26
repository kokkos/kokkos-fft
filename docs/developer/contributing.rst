.. SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
..
.. SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

Contributing
============

We briefly explain the workflow to contribute by making a pull request (PR) against ``main`` branch.
Followings are the steps to contribute the project:

#. **Report bugs or Suggest features:**
   First open issues or ask questions on the Kokkos slack channel.

#. **Fork the Repository:**
   Click the "Fork" button on our `repo <https://github.com/kokkos/kokkos-fft>`_ to create your own copy.

#. **Clone Your Fork:**  
   Clone the repository to your local machine with submodules.

   .. code-block:: bash

      git clone --recursive https://github.com/<your-username>/kokkos-fft.git

#. **Add the Upstream Remote:**  
   Set the original repository as the ``upstream`` remote to keep your fork up to date:

   .. code-block:: bash

      cd kokkos-fft
      git remote add upstream https://github.com/kokkos/kokkos-fft.git   

#. **Create a New Branch:**
   Create a new branch for your feature or bug fix with a descriptive name:

   .. code-block:: bash

      git switch -c <feature-name>

#. **Rebase Your Branch Against Upstream/main:**  
   To ensure your branch is up to date with the latest changes from the original repository:

   1. Fetch the latest changes from upstream:

      .. code-block:: bash

         git fetch upstream main

   2. Rebase your branch onto the upstream ``main`` branch:

      .. code-block:: bash

         git rebase upstream/main

#. **Make Your Changes:**  
   Implement your feature or bug fix. Ensure you adhere to the project's coding standards and guidelines (see :doc:`details of CI<CI>`).

#. **Commit Your Changes:**  
   Stage your changes and commit them with a clear, concise commit message:

   .. code-block:: bash

      git add .
      git commit -m "Brief description of changes"

#. **Push Your Branch:**  
   Push your branch to your GitHub fork:

   .. code-block:: bash

      git push --force-with-lease origin <feature-name>

#. **Open a Pull Request:**  
   On `GitHub <https://github.com/kokkos/kokkos-fft>`_, open a PR from your branch to the original repository's ``main`` branch.
   Include a detailed description of your changes and reference any related issues.

#. **Participate in the Code Review:**  
   Respond to any feedback or questions from maintainers. To run unit-tests on GPUs, you need a special CI approval from maintainers.
   Update your PR as needed until it meets the project requirements.

#. **Merge Your Changes:**  
   Once your pull request is approved, your changes will be merged into the main project. Thank you for your contribution!
