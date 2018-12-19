MPICC	= mpicxx
NVCC    = nvcc
NVFLAGS = -Xptxas --opt-level=3 -arch sm_30 -Xcompiler -fopenmp -Xcompiler -O3 
LDFLAGS = -L/usr/local/cuda/lib64 -L/usr/local/cuda/include -fopenmp -lcudart -lmpi -lm
CFLAGS  = -I/usr/lib/openmpi/include -L /usr/lib/openmpi/lib

main: main.o load_balance.o calculus.o gpu_calculus.o omp_calculus.o seq_calculus.o
	$(MPICC) -o $@ $^ $(LDFLAGS)

main.o: main.cu
	$(NVCC) $(CFLAGS) -c $<

load_balance.o: load_balance.cu load_balance.h
	$(NVCC) $(CFLAGS) -c $<

calculus.o: calculus.cu calculus.h
	$(NVCC) $(NVFLAGS) -c $<

gpu_calculus.o: gpu_calculus.cu gpu_calculus.h
	$(NVCC) $(NVFLAGS) -c $<

omp_calculus.o: omp_calculus.cu omp_calculus.h
	$(NVCC) $(NVFLAGS) -c $<

seq_calculus.o: seq_calculus.cu seq_calculus.h
	$(NVCC) $(NVFLAGS) -c $<

clean:
	rm -f *.o main