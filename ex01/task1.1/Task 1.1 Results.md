# Task 1.1 Results

| copy type  | Host to Device Bandwidth | Device to Host Bandwidth | Device to Device Bandwidth |
| ---------- | ------------------------ | ------------------------ | -------------------------- |
| cudaMemcpy | 12.5                     | 13                       | 313.7                      |
| cudaMemcpy | 12.7                     | 13                       | 313.8                      |
| cudaMemcpy | 12.7                     | 13                       | 313.5                      |
| Average    | 12.63333                 | 13                       | 313.6667                   |

| copy type  | Host to Device Bandwidth | Device to Host Bandwidth | Device to Device Bandwidth |
| ---------- | ------------------------ | ------------------------ | -------------------------- |
| copyKernel | 12.3                     | 11.8                     | 166.7                      |
| copyKernel | 12.3                     | 12                       | 164.8                      |
| copyKernel | 12.3                     | 11.9                     | 165                        |
| Average    | 12.3                     | 11.9                     | 165.5                      |

As it shows in the tables, the difference between `cudaMemcpy` and `copyKernel` is relatively small in H2D, and D2H. `copyKernel` might be slightly slower than `cudaMemcpy` in these 2 cases. However, in D2D situation, `copyKernel` is significantly  (about 47%) slower than `cudaMemcpy`.