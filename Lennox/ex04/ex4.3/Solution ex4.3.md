# Solution ex4.3

1. Depending on `warpNum`, how many additions are performed in the original and in the new version? 

    For the original version:  

    ```
    addition times = 
    blockNum*(warpNum * 63 + (2*warpNum-1) )
    + (blockNum * 63 + (2*blockNum-1)) 
    + ((blockNum-1)/blockNum *N)
    
    with blockNum= N / 256
    ```
    
    For the new version:
    
    ```
    addition times = 
    blockNum*(32* (2*warpNum-1) + 63 )
    + (blockNum * 63 + (2*blockNum-1))  
    + ((blockNum-1)/blockNum *N)
    
    with blockNum= N / 256
    ```

2. Does the new version produce bank conflicts, can we avoid them?

     For the new version, loading data to shared memory does not produce bank conflict. However, when making serial summations , it might produce bank conflicts, if warpNum is an integer multiple of 32. That's because the lanes are accessing the data in the same column, with the interval of length of row warpNum. When warpNum is an integer multiple of 32, the lanes will access the same memory bank.

    We can avoid them by making warp_num not a integer multiple of  32.

3. What limits performance in the new version?

    The performance is limited by the size of the share memory, which limits the number of elements that can be process within one block.