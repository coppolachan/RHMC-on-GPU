#ifndef RATIONALAPPROX_H_
#define RATIONALAPPROX_H_

#include "include/global_const.h"

class ShiftMultiFermion;
class MultiFermion;

//                                                                    __ i<approx_order       RA_a[i]
//  approximation valid in [min_epsilon, 1] of the form  f(x)=RA_a0 + \                  --------------------
//                                                                    /_ i=0                x + RA_b[i]

class RationalApprox {
private:
 REAL min_epsilon;
 int approx_order;
 REAL RA_a0;
 REAL RA_a[max_approx_order];
 REAL RA_b[max_approx_order];
public:
 RationalApprox();
 RationalApprox(const REAL eps, const int order, const REAL a0, const REAL *a, const REAL *b);
 RationalApprox& operator=(const RationalApprox &rhs); 

 friend void get_order(int &order, RationalApprox approx);
 friend void get_shifts(float shifts[max_approx_order], RationalApprox approx);
 friend void get_shifts(double shifts[max_approx_order], RationalApprox approx);
 friend void get_const(float &a, RationalApprox approx);
 friend void get_const(double &a, RationalApprox approx);
 friend void get_numerators(float numerators[max_approx_order], RationalApprox approx);
 friend void get_numerators(double numerators[max_approx_order], RationalApprox approx);
 friend std::ostream& operator<<(std::ostream &os, const RationalApprox &approx); 

 // rescaled coefficients
 void first_inv_approx_coeff(void);
 void md_inv_approx_coeff(void);
 void last_inv_approx_coeff(void);

 friend void first_inv_approx_calc(REAL res);
 friend void last_inv_approx_calc(REAL res);

 // defined in Inverter/inverter.cc
 friend void multips_shifted_invert(ShiftMultiFermion *chi, MultiFermion *phi, REAL res, RationalApprox approx);

 // defined in FermionForce/fermionforce.cc
 friend void fermionforce(void);
};

void first_inv_approx_calc(REAL res);
void last_inv_approx_calc(REAL res);

#endif //RATIONALAPPROX_H_
