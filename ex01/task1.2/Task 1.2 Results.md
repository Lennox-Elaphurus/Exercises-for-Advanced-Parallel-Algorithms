# Task 1.2 Results



| copy size | number of transfer  instructions     per thread | Host to Device Bandwidth | Device to Host Bandwidth | Device to Device Bandwidth | Transfer Size (Bytes) |
| --------- | ----------------------------------------------- | ------------------------ | ------------------------ | -------------------------- | --------------------- |
| 4         | 1                                               | 12.4                     | 8.2                      | 279                        | 100000000             |
| 4         | 2                                               | 12.5                     | 8.4                      | 263.8                      | 100000000             |
| 4         | 4                                               | 11.6                     | 8.1                      | 265.8                      | 100000000             |
| 4         | 8                                               | 9.3                      | 7.8                      | 276.2                      | 100000000             |
| 8         | 1                                               | 12.5                     | 8.5                      | 259.2                      | 100000000             |
| **8**     | **2**                                           | **10.5**                 | **8.1**                  | **288**                    | 100000000             |
| 8         | 4                                               | 9.5                      | 8                        | 251.1                      | 100000000             |
| 8         | 8                                               | 8.9                      | 7.7                      | 255.9                      | 100000000             |
| 16        | 1                                               | 9.8                      | 8.3                      | 246.8                      | 100000000             |
| 16        | 2                                               | 9.4                      | 8.1                      | 271.3                      | 100000000             |
| 16        | 4                                               | 9.1                      | 7.8                      | 277                        | 100000000             |
| 16        | 8                                               | 8.7                      | 7.6                      | 268.9                      | 100000000             |

The result of D2D bandwidth is unstable. In this case, `copy size = 8, number of transfer instrunctions per thread = 2` obtained the best performance.

.