// Defines a toolbox

#ifndef TOOLS_H_
#define TOOLS_H_

#include <string>
#include "include/global_macro.h"

using namespace std;

// existence of files
int look_for_file(string filename); 


// random number generator
REAL uniform_generator(void);
void initrand(unsigned long s);
void su2_rand(REAL &p0, REAL &p1, REAL &p2, REAL &p3);

// rational approximation
void rationalapprox_calc(void);

//global sums
//G++ requires template header in the same file as implementation. In practice it means implementation inside .h file.
template<typename T> void global_sum(T *vec, long int i)
{
  if(i==1)
    {
    // base case
    }
  else
    {
    if(i%2==0)
      {
      long int j, k;

      j=i/2;
      for(k=0; k<j; k++)
         {
         vec[k]+=vec[k+j];
         }
      global_sum(vec, j);
      }
    else
      {
      long int j1, j2, k;
  
      j1=i/2;
      j2=j1+1;
      for(k=0; k<j1; k++)
         {
         vec[k]+=vec[k+j2];
         }
      global_sum(vec, j2);
      }
    }
  }


#endif //TOOLS_H_


