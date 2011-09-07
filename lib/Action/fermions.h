#ifndef FERMION_H_
#define FERMION_H_

#include "include/global_var.h"
#include "lib/Tools/vec3.h"
#include "lib/Tools/RationalApprox/rationalapprox.h"


// SINGLE FERMION
class Fermion {
private:
 Vec3 fermion[sizeh];
public:
 Fermion(void);
 void gauss(void);
 void z2noise(void);
 double l2norm2(void);

 // defined below 
 friend void extract_fermion(Fermion *out, MultiFermion *in, int i);

 // defined in FermionForce/fermionforce.cc
 friend void fermionforce(void);

 // defined in FermionMatrix/fermionmatrix.cc
 friend void Deo(Fermion *out, const Fermion *in);
 friend void Doe(Fermion *out, const Fermion *in);

 // defined in Findminmax/findminmax.cc
 friend void findmaxeig(REAL &max);
 friend void findmineig(REAL &min, const REAL &max);

 // defined in Inverter/inverter.cc
 friend void invert (Fermion *phi, Fermion *chi, REAL res);
 friend void multips_shifted_invert (ShiftMultiFermion *chi, MultiFermion *phi, REAL res, RationalApprox approx);

 // defined in Inverter/cu_inverter.cc
 friend void cu_multips_shifted_invert (REAL res, RationalApprox approx);

 // defined in Meas/chiralmeas.cc
 friend void chiral_meas(complex<REAL> &chiral, complex<REAL> &e_dens, complex<REAL> &b_dens, complex<REAL> &p_den);

 // defined in Packer/packer.cc
 friend void smartpack_fermion(float *out, const Fermion *in);
 friend void smartpack_fermion_d(float out[6*sizeh*2], const Fermion *in);
 friend void smartunpack_fermion_d(Fermion *out, const float in[6*sizeh*2]);
};



//SHIFT FERMIONS
class ShiftFermion {
private:
 Vec3 fermion[max_approx_order][sizeh];
public:
 ShiftFermion(void);

 // defined below 
 friend void extract_fermion(ShiftFermion *out, ShiftMultiFermion *in, int i);

 // defined in Inverter/inverter.cc
 friend void multips_shifted_invert (ShiftMultiFermion *chi, MultiFermion *phi, REAL res, RationalApprox approx);
};








// MULTIPLE FERMIONS
class MultiFermion {
private:
  Vec3 **fermion;//[no_ps][sizeh];
public:
  MultiFermion(void);
  
  friend void create_phi(void);
  friend void extract_fermion(Fermion *out, MultiFermion *in, int i);
  
  // defined in Action/action.cc
  friend void fermion_action(double *value, int init);
  
  // defined in FermionForce/fermionforce.cc
  friend void fermionforce(void);

  // defined in Inverter/inverter.cc
  friend void multips_shifted_invert (ShiftMultiFermion *chi, MultiFermion *phi, REAL res, RationalApprox approx);
  
  // defined in Inverter/cu_inverter.cc
  friend void cu_multips_shifted_invert (REAL res, RationalApprox approx);
  
  // defined in RationalApprox/rationalapprox.cc
  friend void first_inv_approx_calc(REAL res);
  friend void last_inv_approx_calc(REAL res);
  
  // defined in Packer/packer.cc
  friend void smartpack_multifermion(float *out , const MultiFermion *in);
  friend void smartunpack_multifermion(MultiFermion *out, const float *in);
};

// Initialize fermion_phi with gaussian random numbers
void create_phi(void);






// SHIFT MULTIPLE FERMIONS
class ShiftMultiFermion {
private:
 Vec3 fermion[no_ps][max_approx_order][sizeh];
public:
 ShiftMultiFermion(void);

 friend void extract_fermion(ShiftFermion *out, ShiftMultiFermion *in, int i);

 // defined in FermionForce/fermionforce.cc
 friend void fermionforce(void);

 // defined in Inverter/inverter.cc
 friend void multips_shifted_invert (ShiftMultiFermion *chi, MultiFermion *phi, REAL res, RationalApprox approx);

 // defined in Inverter/cu_inverter.cc
 friend void cu_multips_shifted_invert (REAL res, RationalApprox approx);

 // defined in RationalApprox/rationalapprox.cc
 friend void first_inv_approx_calc(REAL res);
 friend void last_inv_approx_calc(REAL res);

 // defined in Packer/packer.cc
 friend void smartunpack_multishiftfermion(ShiftMultiFermion *out, const float *in,  int order);
};


#endif //FERMION_H_
