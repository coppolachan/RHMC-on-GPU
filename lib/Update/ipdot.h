#ifndef IPDOT_H_
#define IPDOT_H_

#include "lib/Action/su3.h"

// defined in FermionForce/fermionforce.cc
//void fermionforce(void);

class Ipdot {
private:
  Su3 ipdot[no_links];
public:
  Ipdot(void);

  friend void calc_ipdot_gauge(void);
  friend void calc_ipdot_fermion(void);
  friend void calc_ipdot(void);

  // defined in FermionForce/fermionforce.cc
  friend void fermionforce(void);

  // defined in Momenta/momenta.cc
  friend void momenta_sum_multiply(const complex<REAL> p);

  // defined in Packer/packer.cc
  friend void smartpack_tamatrix(float out[8*no_links], Ipdot *in);
  friend void smartunpack_tamatrix(Ipdot *out, float in[8*no_links]);
};


// calculate the complete ipdot
void calc_ipdot(void);

// calculate the gauge part of ipdot
void calc_ipdot_gauge(void);

// calculate the fermionic part of ipdot
void calc_ipdot_fermion(void);

#endif //IPDOT_H_
