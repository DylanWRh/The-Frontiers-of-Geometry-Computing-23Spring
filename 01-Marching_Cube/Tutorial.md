# Assignment 1

## TODO

- Read data (128x128x128, float32) from SDF files in [data](./data).
- Implement the Marching Cubes algorithm and extract the zero isosurface.
- Save the results as OBJ/OFF/PLY format.
- Write a report.

## Requirements

- Except for the algorithm, I/O operation should also be implemented by yourself.
- Programming language can be Python/C++.
- Mind your code style.

## Hint

- OBJ/OFF/PLY file can be visualized with **meshlab** or [**meshviewer**](../MeshViewer) attached.
- You are free to search online for the mapping tables needed and directly use them. But the other part of the algorithm should be implemented independently.
- For Python, you can use `numpy.fromfile()` to read the binary files. Note that you need to concat the data in two files to get the full data.
