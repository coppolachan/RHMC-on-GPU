#include <iostream>
#include "lib/Tools/exception.h"

#include "include/global_const.h"

void test_param(void)
  {
  #ifdef DEBUG_MODE
  cout << "DEBUG: inside test_param..."<<endl;
  #endif

  if(nx%2 + ny%2 +nz%2 +nt%2 >0)
    {
    throw odd_sides;
    }

  #ifdef PURE_GAUGE
  if(use_multistep!=0) throw no_multistep;
  #endif     

  #ifdef USE_GPU
  if(size % (2*NUM_THREADS)!=0) throw wrong_num_threads;
  #endif

  #ifdef DEBUG_MODE
  cout << "\tterminated test_param..."<<endl;
  #endif
  }
