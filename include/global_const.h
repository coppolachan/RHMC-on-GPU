//Definition of global constants

#ifndef GLOBAL_CONST_H_
#define GLOBAL_CONST_H_

#include <complex>
#include "include/parameters.h"

using namespace std;

const int max_approx_order=(approx_metro>approx_md) ? approx_metro : approx_md;

// derived mass constants
const REAL one_by_mass=1.0/mass;
const REAL mass2=mass*mass;

// derived beta constants
const REAL beta_by_three=beta/3.0;

// derived lattice related constants
const long int vol1=nx;
const long int vol2=ny*vol1;
const long int vol3=nz*vol2;
const long int vol4=nt*vol3;

const long int no_links=4*vol4;

const long int size=vol4;
const long int size2=2*size;
const long int size3=3*size;

const long int sizeh=size/2;
const long int size3h=3*sizeh;
const long int size5h=5*sizeh;
const long int size7h=7*sizeh;

const REAL inv_size=1.0/(REAL) size;
const REAL inv_size_by_three=inv_size/3.;
const REAL inv_vol3=1.0/(REAL) vol3;

#ifdef IM_CHEM_POT
const REAL eim_cos=cos(immu);
const REAL eim_sin=sin(immu);
//  #ifndef __CUDACC__  // if we are not inside nvcc
//  const complex<REAL> eim=complex<REAL>(eim_cos, eim_sin);
//  const complex<REAL> emim=complex<REAL>(eim_cos, -eim_sin);
//  #endif
#endif

// molecular dynamic related derived constants
#ifndef __CUDACC__  // if we are not inside nvcc
const REAL epsilon=1.0/((REAL) no_md);
const complex<REAL> ieps=complex<REAL>(0.0, epsilon);
const complex<REAL> iepsh=ieps*complex<REAL>(0.5,0.0);
#endif

// mathematical constants
const REAL one_by_three=0.3333333333333333333333333333333333333333333333333333333333333333333;
const REAL one_by_sqrt_two=0.7071067811865475244008443621048490392848359376884740365883398689953;
const REAL one_by_sqrt_three=0.5773502691896257645091487805019574556476017512701268760186023264839;
const REAL two_by_sqrt_three=2.0*one_by_sqrt_three;
const REAL pi=3.141592653589793238462643383279502884197169399375105820974944;
const REAL pi2=6.283185307179586476925286766559005768394338798750211641949889;
const REAL half_pi=1.570796326794896619231321691639751442098584699687552910487472;

const float min_value_float=5.0e-5;
const double min_value_double=1.0e-13;

#define _MIN_VALUE(x) min_value##_##x
#define MIN_VALUE(x) _MIN_VALUE(x)
// in this way MIN_VALUE(REAL) is min_value_float or min_value_double dependig on the value of REAL


#endif //GLOBAL_CONST_H_
