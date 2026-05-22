all:
	nvcc tensorsExample.cu -lineinfo -lcublas -lcurand -o tensorsExample
