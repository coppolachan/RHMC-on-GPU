#ifndef CONFIGURATION_H_
#define CONFIGURATION_H_

#include "lib/Action/su3.h"


class Conf {
 private:
  Su3 u_save[no_links];
  Su3 u_work[no_links];
  
  // base constructor private!
  Conf(void);
 public:
  Conf(int init);
  
  void save(void);
  void copy_saved(void);
  void write(void);
  void write_last(void);
  void print(void);
  
  void unitarize_with_eta(void);

  // defined in Action/action.cc
  void loc_action(double *value, int init) const; 

  // defined in FermionMatrix/fermionmatrix.cc
  friend void Deo(Fermion *out, const Fermion *in);
  friend void Doe(Fermion *out, const Fermion *in);

  // defined in Ipdot/ipdot.cc
  friend void calc_ipdot_gauge(void);
  friend void calc_ipdot_fermion(void);
  friend void calc_ipdot(void);

  // defined in Meas/gaugemeas.cc
  void calc_plaq(REAL &pls, REAL &plt) const;
  void calc_poly(REAL &re, REAL &im) const ;

  // defined in Meas/chiralmeas.cc
  friend void chiral_meas(complex<REAL> &chiral, 
			  complex<REAL> &e_dens, 
			  complex<REAL> &b_dens, 
			  complex<REAL> &p_den);

  // defined in Momenta/momenta.cc
  friend void conf_left_exp_multiply(const complex<REAL> p);

  // defined in Staples/staples.cc
  friend void calc_staples(void);

  // defined in Packer/packer.cc
  friend void smartpack_gauge(float out[2*12*no_links] , const Conf *in);
  friend void smartunpack_gauge(Conf *out, const float in[2*12*no_links]);

  // defined in RevUpdate/revupdate.cc
  #ifdef REVERSIBILITY_TEST
    friend void rev_update(void);
  #endif
};


#endif //CONFIGURATION_H_
