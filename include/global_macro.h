#ifndef GLOBAL_MACRO_H_
#define GLOBAL_MACRO_H_


#define REAL double

// number of threads if GPU is used
#define NUM_THREADS 128

//#define REVERSIBILITY_TEST
//#define PARAMETER_TEST

//#define PURE_GAUGE

//#define NO_MEASURE
#ifdef NO_MEASURE
 #warning "No measures will be taken!!"
#endif

//#define DEBUG_MODE
//#define DEBUG_MODE_2
//#define DEBUG_INVERTER

//#define TIMING_CUDA_CPP

#if (!defined REVERSIBILITY_TEST) && (!defined PARAMETER_TEST)
  #define STANDARD_UPDATE
#endif

#define CONF_FILE config
#define DATA_FILE data.dat
#define ERROR_FILE data.err

// if #define val1 val2 then QUOTEME(val1) gives the string "val2"
#define _QUOTEME(x) #x
#define QUOTEME(x) _QUOTEME(x)

// to be used in cuda_err.cu
#define AT __FILE__ " : " QUOTEME(__LINE__)


#endif //GLOBAL_MACRO_H_
