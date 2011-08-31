
#include <complex>
#include "include/global_macro.h"
#include "include/tools.h"
#include "lib/Action/su3.h"

using namespace std;

// gaussian random number generator
REAL real_gauss(void)
   {
   REAL phi, temp, radius, ris;
   
   phi=pi2*uniform_generator();
   temp=-1.0*log(uniform_generator());
   radius=sqrt(temp);

   ris=radius*cos(phi);

   return ris;
   }


// gaussian random number generator
void real_gauss(REAL &r1, REAL &r2)
   {
   REAL phi, temp, radius;
   
   phi=pi2*uniform_generator();
   temp=-1.0*log(uniform_generator());
   radius=sqrt(temp);

   r1=radius*cos(phi);
   r2=radius*sin(phi);
   }


// complex random gaussian number
complex<REAL> complex_gauss(void)
   {
   REAL re, im;

   real_gauss(re, im);

   return complex<REAL>(re, im);
   }


// gaussian disrtributed matrix in Su3 algebra:  a_{i-1}*lambda_i , a_{i-1} with normal distribution
//
//            | 0  1  0 |            | 0 -i  0 |            | 1  0  0 |
//  lambda_1= | 1  0  0 |  lambda_2= | i  0  0 |  lambda_3= | 0 -1  0 |
//            | 0  0  0 |            | 0  0  0 |            | 0  0  0 |
//
//            | 0  0  1 |            | 0  0 -i |            | 0  0  0 |
//  lambda_4= | 0  0  0 |  lambda_5= | 0  0  0 |  lambda_6= | 0  0  1 |
//            | 1  0  0 |            | i  0  0 |            | 0  1  0 |
//
//            | 0  0  0 |            | 1  0  0 |
//  lambda_7= | 0  0 -i |  lambda_8= | 0  1  0 |*(1/sqrt(3))
//            | 0  i  0 |            | 0  0 -2 |
//
void Su3::gauss(void)
  {
  int i;
  REAL rand[8], aux1, aux2;

  for(i=0; i<4; i++)
     {
     real_gauss(aux1, aux2);
     rand[2*i]=aux1;
     rand[2*i+1]=aux2;
     }

  comp[0][0]=complex<REAL>(rand[2] + one_by_sqrt_three*rand[7], 0.0);
  comp[0][1]=complex<REAL>(rand[0], -rand[1]);
  comp[0][2]=complex<REAL>(rand[3], -rand[4]);

  comp[1][0]=complex<REAL>(rand[0], rand[1]);
  comp[1][1]=complex<REAL>(-rand[2] + one_by_sqrt_three*rand[7], 0.0);
  comp[1][2]=complex<REAL>(rand[5], -rand[6]);
 
  comp[2][0]=complex<REAL>(rand[3], rand[4]);
  comp[2][1]=complex<REAL>(rand[5], rand[6]);
  comp[2][2]=complex<REAL>( -two_by_sqrt_three * rand[7], 0.0);
  }


// Complex gaussian vector
void Vec3::gauss(void)
  {
  comp[0]=complex_gauss();
  comp[1]=complex_gauss();
  comp[2]=complex_gauss();
  }

// Z2 noise vector
void Vec3::z2noise(void)
  {
  double p;
  
  p=uniform_generator();
  if(p<0.5) comp[0]=complex<REAL>(1.0,0.0);
  else  comp[0]=complex<REAL>(-1.0,0.0);

  p=uniform_generator();
  if(p<0.5) comp[1]=complex<REAL>(1.0,0.0);
  else  comp[1]=complex<REAL>(-1.0,0.0);

  p=uniform_generator();
  if(p<0.5) comp[2]=complex<REAL>(1.0,0.0);
  else  comp[2]=complex<REAL>(-1.0,0.0);
  }
