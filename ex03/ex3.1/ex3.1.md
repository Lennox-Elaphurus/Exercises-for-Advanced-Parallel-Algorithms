# ex3.1

## （1）

`__shfl_sync(unsigned mask, T var, int srcLane, int width=warpSize);` gives the value from high ID lane to low ID lane. Because if `srcLane` is outside the range `[0:width-1]`, the value returned corresponds to the value of var held by the `srcLane modulo width`. In particular, we can let `srcLane = laneID - delta;` then when `srcLane < 0`, it will get the value from the highest ID lane.

## (2)

If we were on a compute capability 8.x  or higher GPU, we can simply use `int __reduce_min_sync(unsigned mask, int value);` to get the minimum value of `value`, then use `unsigned int __match_any_sync(unsigned mask, T value);` to find the index of the lane which holds the minimum value.

On  a compute capability 7.x GPU, to get the minimum value of `value`, we can:

1. use `__shfl_xor_sync`to get the value of a neighboring lane
2. compare the `value` in this lane and the received `value`, and store the smaller value for the next turn.
3. decrease the size of active lanes by factor of 2 
4. repeat 1.2.3. until we get a single value, which is the minimum value

Then also use `__match_any_sync` to get the lane ID of lane which holds the minimum value in the first turn.