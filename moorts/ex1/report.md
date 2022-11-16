# Exercise 1.3

We implemented a templated kernel that works for multiple numeric types as type parameters. The kernel differentiates the template type and uses the appropriate vector types (int2 for integers, float2 for floating points) to work on the ideal size of bytes at the same time (8).

E.g. for integers, we scaled two consecutive integers independantly and then combined them into an int2 value, that we then assigned to the destination array. This does not seem like the intended approach, but we didn't find functions that allow scalar multiplication of vector types.

Our kernel also uses the ranges from 1.2 to execute a fixed number of instructions per thread.
