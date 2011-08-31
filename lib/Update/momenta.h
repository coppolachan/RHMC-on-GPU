#ifndef MOMENTA_H_
#define MOMENTA_H_

#include "include/global_macro.h" //defines REVERSIBILITY_TEST
#include "lib/Action/su3.h"

class Momenta {
 private:
  Su3 momenta[no_links];
 public:
  Momenta(void);
  
  friend void create_momenta(void);
  friend void conf_left_exp_multiply(const complex<REAL> p);
  friend void momenta_sum_multiply(const complex<REAL> p);
  
  // defined in Packer/packer.cc
  friend void smartpack_thmatrix(float out[8*no_links], Momenta *in);
  friend void smartunpack_thmatrix(Momenta *out, float in[8*no_links]);
  
  // defined in Update/action.cc
  friend void calc_action(double *value, int init);
  
#ifdef REVERSIBILITY_TEST
  friend void reverse_momenta(void);
  friend void save_momenta(void);
  friend void rev_update(void);
#endif
};

void create_momenta(void);

// u_work[i] -> exp(p*momenta[i])*u_work[i]
void conf_left_exp_multiply(const complex<REAL> p);

// momenta[i]+=p*ipdot_loc[i]
void momenta_sum_multiply(const complex<REAL> p);

#ifdef REVERSIBILITY_TEST
void reverse_momenta(void);
void save_momenta(void);
#endif

#endif //MOMENTA_H_
