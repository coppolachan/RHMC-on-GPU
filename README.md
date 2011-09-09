Rational Hybrid Montecarlo with non improved staggered fermions
===============================================================

---
**Developers and mantainers**  
*Claudio Bonati* (C++ framework, CUDA kernels) (<bonati@df.unipi.it>)  
*Guido Cossu* (CUDA kernels) (<cossu@post.kek.jp>)  

*History*:  
**Version 0.1**  9/2011     CUDA kernels

---

## 1. Features 


Performs Rational Hybrid Monte Carlo simulation of QCD with  N_f flavors
of staggered (non improved) fermions. 
Except some minor functions, all the calculations are performed using
NVIDIA gpus, implementing a CUDA version of all the relevant kernels 
(Dirac operator, Dirac operator inversion, gauge and momenta update).

See the following reference for more details:  
*'QCD simulations with staggered fermions on GPUs'*  
[Arxiv Paper 1106.5673](http://inspirebeta.net/record/916054)


## 2. Compilation


The typical compilation follows the pattern:

`./configure && make`

this will compile a code to run in a single core CPU.

### 3. Enabling CUDA routines

In order to enable CUDA routines type the following command (also look at the 
various options in `configure --help`)

`./configure --enable-cuda=ARCH`

where ARCH should be substituted by one of the following strings "sm_XX" 
in order to choose the target NVIDIA architecture

*  *sm_10*  ISA_1, Basic features  
*  *sm_11*  atomic memory operations on global memory  
*  *sm_12*  atomic memory operations on shared memory, vote instructions  
*  *sm_13* double precision floating point support (Tesla Cards)    
*  *sm_20*  Fermi architecture support  

please refer to the original [NVIDIA documentation](http://developer.nvidia.com/cuda-toolkit-40) for more details 
on which cards support the several architectures.

## 4. Issues


Some newer compilers, like GNU `gcc 4.5`, are incompatible with the NVIDIA `nvcc`
compiler (CUDA 4.0). In order to solve this problem you should install a different 
version of the compiler, e.g. `gcc 4.4`, in the directory `PATH_TO_COMPILER` and then
configure with the following command

`./configure --enable-cuda=ARCH --with-cuda_comp=PATH_TO_COMPILER 
CXX=PATH_TO_COMPILER/g++ CC=PATH_TO_COMPILER/gcc`

Also, if the mandatory library [MPFR](http://www.mpfr.org/) (for Multiple Precision in rational approximations) is 
installed in a non standard location (not `/usr/lib`) use the flag `--with-mpfr=PATH` where `PATH` is the location
of the library in your system

## 5. Known bugs

 * Some older versions of `autoreconf` fail to build a working `configure` file. Usually the one shipped is working.

If you find any bug please report any to the developers on GitHub issues:

[RHMC on GPU issues](https://github.com/coppolachan/RHMC-on-GPU/issues)

