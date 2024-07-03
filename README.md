# NLP Sampling

This code is accompanying the paper *NLP Sampling: Notes on Combining
MCMC and NLP Methods for Diverse Constrained Sampling*.

The repository contains the exact snapshot with which all experiments were
done. The actual NLP_Sampler is part of the rai repo, and the robot
models are in the rai-robotModels. Both have to be checked out in
parallel (to a specific commit, to make it exactly reproducible):

```
git clone -b 8547d4380a2c5169f63d6fe18b6d234143cc59c4 https://github.com/MarcToussaint/rai
git clone -b 303ff9aa51508f5e765c99af552b485ecb4586df https://github.com/MarcToussaint/rai-robotModels
git clone https://github.com/MarcToussaint/24-NLP-Sampling
```

* `make -j1 -C rai installUbuntuAll` should install necessary Ubuntu
 packages
* `cd 24-NLP-Sampling && make -j $(command nproc --ignore 2)` should compile everything
* The python scripts demos.py and run.py generate the results
* If you have the python package robotic installed, you can check
  `ry-view scene.g`
* If compiling is a mess, there is also a way to link to the pre-compiled
  lib that ships with the robotic python package (let me know and I'll
  provide instructions for this).
