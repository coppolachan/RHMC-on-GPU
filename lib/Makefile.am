# Include Path
CXXFLAGS += $(CUDA_COMPILE) $(MPFR_INCFLAGS) -Wall --pedantic
INCFLAGS = -I. -I$(top_srcdir)
LDFLAGS  += $(CUDA_LIBS) $(MPFR_LIB)
#LDFLAGS  += 

AM_CPPFLAGS=-I$(top_srcdir)
AM_CFLAGS  =-I$(top_srcdir)


SUFFIXES = .cu

.cu.$(OBJEXT):
	$(NVCC) $(INCFLAGS) -o $@ -c $(DEFS) $< $(NVCCFLAGS)

.cu$(EXEEXT):
	$(NVCC) $(INCFLAGS) -o $@  $< $(NVCCFLAGS)

distdir = $(prefix)
#inputdir = $(prefix)/input
#dist_input_DATA = input


bin_PROGRAMS = rhmc
if ENABLE_CUDA
CUDA_SRCS = ./Cuda/DeviceInfo/device_info.cu		\
	./Cuda/cuda_init.cu				


TMP_DIST = test_device
test_device_SOURCES = ./Cuda/DeviceInfo/test_device.cu
endif

rhmc_SOURCES = 	./Main/rhmc.cc		\
	./Main/init.cc			\
	./Main/Geometry/geometry.cc	\
	./Main/Gpu/gpu.cc		\
	./Main/Conf/conf.cc		\
	./Update/update.cc		\
	./Update/momenta.cc		\
	./Action/su3.cc			\
	./Action/action.cc		\
	./Action/fermionforce.cc	\
	./Action/fermionmatrix.cc	\
	./Action/fermions.cc		\
	./Action/staples.cc		\
	./Action/Inverter/inverter.cc 	\
	./Action/Inverter/cu_inverter.cc\
	./Action/Findminmax/findminmax.cc		\
	./Update/ipdot.cc				\
	./Update/MD_integrators/leapfrog.cc		\
	./Update/MD_integrators/minimum_norm2.cc	\
	./Update/MD_integrators/multistep_2MN.cc	\
	./Update/MD_integrators/multistep_4MN.cc	\
	./Update/RevUpdate/revupdate.cc			\
	./Measurements/chiralmeas.cc			\
	./Measurements/gaugemeas.cc			\
	./Tools/RationalApprox/rationalapprox_calc.cc 	\
	./Tools/RationalApprox/rationalapprox.cc	\
	./Tools/RationalApprox/alg_remez.cc		\
	./Tools/exception.cc				\
	./Tools/exist_file.cc				\
	./Tools/gauss.cc				\
	./Tools/Packer/packer.cc			\
	./Tools/Rand/random.cc				\
	./Tools/Rand/RANDOM/dSFMT.c			\
	./Tools/vec3.cc					\
	./Tools/InputParser/InputParser.cc		\
	./Tools/InputParser/InputTextParser.cc		\
	$(CUDA_SRCS)

if ENABLE_CUDA
dist_PROGRAMS = $(TMP_DIST) 
endif

include_HEADERS = 	$(top_srcdir)/include/configuration.h	\
	$(top_srcdir)/include/cu_inverter.h			\
	$(top_srcdir)/include/geometry.h			\
	$(top_srcdir)/include/global_const.h			\
	$(top_srcdir)/include/global_macro.h			\
	$(top_srcdir)/include/global_var.h			\
	$(top_srcdir)/include/gpu.h				\
	$(top_srcdir)/include/init.h				\
	$(top_srcdir)/include/parameters.h			\
	$(top_srcdir)/include/rhmc_include.h			\
	$(top_srcdir)/include/tools.h				\
	$(top_srcdir)/include/update.h				\
	./Action/action.h				\
	./Action/fermionforce.h				\
	./Action/fermions.h				\
	./Action/staples.h				\
	./Action/su3.h					\
	./Action/Findminmax/findminmax.h		\
	./Update/momenta.h				\
	./Update/ipdot.h				\
	./Update/MD_integrators/md_integrators.h	\
	./Update/RevUpdate/revupdate.h			\
	./Tools/RationalApprox/alg_remez.h		\
	./Tools/RationalApprox/bigfloat.h		\
	./Tools/RationalApprox/rationalapprox.h		\
	./Tools/exception.h				\
	./Tools/Rand/RANDOM/dSFMT.h			\
	./Tools/Rand/RANDOM/dSFMT-params.h		\
	./Tools/Rand/RANDOM/dSFMT-params11213.h		\
	./Tools/Packer/packer.h				\
	./Tools/vec3.h					\
	./Tools/InputParser/InputParser.h		\
	./Tools/InputParser/InputTextParser.h		\
	./Tools/InputParser/StringTools.h		\
	./Cuda/cuda_find_min_max.h			\
	./Cuda/cuda_deriv.h				\
	./Cuda/cuda_exponentiate.h			\
	./Cuda/cuda_fermion_force.h			\
	./Cuda/cuda_init.h				\
	./Cuda/cuda_inverter.h				\
	./Cuda/cuda_rhmc.cuh				\
	./Cuda/cuda_mom_sum_mul.h			\
	./Cuda/cuda_sol_sum.h				\
	./Cuda/cuda_reduce.h				\
	./Cuda/cuda_rhmc.cuh				\
	./Cuda/cuda_err.cuh				\
	./Cuda/DeviceInfo/device_info.h			\
	./Cuda/cuda_dslash_kernels.h			\
	./Cuda/cuda_inversion_kernels.h			\
	./Cuda/cuda_tool_kernels.h			\
	./Cuda/cuda_update_kernels.h

EXTRA_DIST = ./Cuda/cuda_deo.cu				\
	./Cuda/cuda_deriv.cu				\
	./Cuda/cuda_dslash_dd_eo.cu			\
	./Cuda/cuda_dslash_eo.cu			\
	./Cuda/cuda_err.cu				\
	./Cuda/cuda_exponentiate.cu			\
	./Cuda/cuda_fermion_force.cu			\
	./Cuda/cuda_find_min_max.cu			\
	./Cuda/cuda_inverter.cu				\
	./Cuda/cuda_inverter_d.cu			\
	./Cuda/cuda_mom_sum_mul.cu			\
	./Cuda/cuda_reduce.cu				\
	./Cuda/cuda_sol_sum.cu