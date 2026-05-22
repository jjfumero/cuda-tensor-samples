## Example of how to use CUDA Tensor WMMA API 

This test is based on the NVIDIA sample suite for tensors:
- https://github.com/NVIDIA-developer-blog/code-samples/tree/master/posts/tensor-cores.
- https://github.com/NVIDIA/cuda-samples/blob/master/cpp/3_CUDA_Features/cudaTensorCoreGemm/cudaTensorCoreGemm.cu

This test is used to compared the generated CUDA code from [HAT](https://github.com/openjdk/babylon/tree/code-reflection/hat), 
a Java parellel programming framwork to exploit data parallel applications on hardware accelerators, against CUDA native implementations.

### How to build and run?

```bash
$ make
$ ./tensorExample

M = 1024, N = 1024, K = 1024. alpha = 1.000000, beta = 0.000000

Running with CUDA No Tensor WMMA...
Running with Tensor WMMA col_major ...
	grid: 16, 16, 1
	block: 128, 4, 1
Running with Tensor WMMA row_major ...
	grid: 16, 16, 1
	block: 128, 4, 1
Running with cuBLAS...

Checking results...
✅ Results verified: cublas and WMMA agree.

wmma took 0.240768ms
cublas took 0.062624ms

For a faster code using wmma you should check out the cudaTensorCoreGemm sample in the CUDA Toolkit.
This code was written as a demo only!

✅ Results verified: WMMA2 Kernel checked with Naive Impl.

cuda took 1.703040ms
```

