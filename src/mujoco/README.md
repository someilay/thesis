## Building mujoco simulations

* Compile:
```bash
cd <disered folder>
cmake -S . -B build -DMUJOCO_DIR=<path> -DGLFW_DIR=<path>
cd build
make
```
