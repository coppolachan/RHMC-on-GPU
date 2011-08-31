/*

  Mike Clark - 25th May 2005

  bigfloat.h

  Simple C++ wrapper for multiprecision datatype used by AlgRemez
  algorithm

*/

#include <gmp.h>
#include <mpfr.h>

#ifndef INCLUDED_BIGFLOAT_H
#define INCLUDED_BIGFLOAT_H

class bigfloat {

private:

  mpfr_t x;

public:

  bigfloat() { mpfr_init_set_ui(x, 0, mpfr_get_default_rounding_mode()); }
  bigfloat(const bigfloat& y) { mpfr_init_set(x, y.x, mpfr_get_default_rounding_mode()); }
  bigfloat(const unsigned long u) { mpfr_init_set_ui(x, u, mpfr_get_default_rounding_mode()); }
  bigfloat(const long i) { mpfr_init_set_si(x, i, mpfr_get_default_rounding_mode()); }
  bigfloat(const int i) {mpfr_init_set_si(x, (long)i, mpfr_get_default_rounding_mode());}
  bigfloat(const float d) { mpfr_init_set_d(x, (double)d, mpfr_get_default_rounding_mode()); }
  bigfloat(const double d) { mpfr_init_set_d(x, d, mpfr_get_default_rounding_mode()); }  
  bigfloat(const char *str) { mpfr_init_set_str(x, (char*)str, 10, mpfr_get_default_rounding_mode()); }
  ~bigfloat(void) { mpfr_clear(x); }
  operator const double (void) const { return (double)mpfr_get_d(x, mpfr_get_default_rounding_mode()); }

  static void setDefaultPrecision(unsigned long dprec) 
    {
    unsigned long bprec =  (unsigned long)(3.321928094 * (double)dprec);
    mpfr_set_default_prec(bprec);
    }

  void setPrecision(unsigned long dprec) 
    {
    unsigned long bprec =  (unsigned long)(3.321928094 * (double)dprec);
    mpfr_set_prec(x, bprec);
    }
  
  unsigned long getPrecision(void) const { return mpfr_get_prec(x); }

  unsigned long getDefaultPrecision(void) const { return mpfr_get_default_prec(); }

  bigfloat& operator=(const bigfloat& y) 
    {
    mpfr_set(x, y.x, mpfr_get_default_rounding_mode()); 
    return *this;
    }

  bigfloat& operator=(const unsigned long y) 
    { 
    mpfr_set_ui(x, y, mpfr_get_default_rounding_mode());
    return *this; 
    }
  
  bigfloat& operator=(const signed long y) 
    {
    mpfr_set_si(x, y, mpfr_get_default_rounding_mode()); 
    return *this;
    }
  
  bigfloat& operator=(const float y) 
    {
    mpfr_set_d(x, (double)y, mpfr_get_default_rounding_mode()); 
    return *this;
    }

  bigfloat& operator=(const double y) 
    {
    mpfr_set_d(x, y, mpfr_get_default_rounding_mode()); 
    return *this;
    }

  size_t write(void);
  size_t read(void);

  /* Arithmetic Functions */

  bigfloat& operator+=(const bigfloat& y) { return *this = *this + y; }
  bigfloat& operator-=(const bigfloat& y) { return *this = *this - y; }
  bigfloat& operator*=(const bigfloat& y) { return *this = *this * y; }
  bigfloat& operator/=(const bigfloat& y) { return *this = *this / y; }

  friend bigfloat operator+(const bigfloat& x, const bigfloat& y) 
    {
    bigfloat a;
    mpfr_add(a.x, x.x, y.x, mpfr_get_default_rounding_mode());
    return a;
    }

  friend bigfloat operator+(const bigfloat& x, const unsigned long y) 
    {
    bigfloat a;
    mpfr_add_ui(a.x, x.x, y, mpfr_get_default_rounding_mode());
    return a;
    }

  friend bigfloat operator-(const bigfloat& x, const bigfloat& y) 
    {
    bigfloat a;
    mpfr_sub(a.x, x.x, y.x, mpfr_get_default_rounding_mode());
    return a;
    }
  
  friend bigfloat operator-(const unsigned long x, const bigfloat& y) 
    {
    bigfloat a;
    mpfr_ui_sub(a.x, x, y.x, mpfr_get_default_rounding_mode());
    return a;
    }
  
  friend bigfloat operator-(const bigfloat& x, const unsigned long y) 
    {
    bigfloat a;
    mpfr_sub_ui(a.x, x.x, y, mpfr_get_default_rounding_mode());
    return a;
    }

  friend bigfloat operator-(const bigfloat& x) 
    {
    bigfloat a;
    mpfr_neg(a.x, x.x, mpfr_get_default_rounding_mode());
    return a;
    }

  friend bigfloat operator*(const bigfloat& x, const bigfloat& y)  
    {
    bigfloat a;
    mpfr_mul(a.x, x.x, y.x, mpfr_get_default_rounding_mode());
    return a;
    }

  friend bigfloat operator*(const bigfloat& x, const unsigned long y) 
    {
    bigfloat a;
    mpfr_mul_ui(a.x, x.x, y, mpfr_get_default_rounding_mode());
    return a;
    }

  friend bigfloat operator/(const bigfloat& x, const bigfloat& y)
    {
    bigfloat a;
    mpfr_div(a.x, x.x, y.x, mpfr_get_default_rounding_mode());
    return a;
    }

  friend bigfloat operator/(const unsigned long x, const bigfloat& y)
    {
    bigfloat a;
    mpfr_ui_div(a.x, x, y.x, mpfr_get_default_rounding_mode());
    return a;
    }

  friend bigfloat operator/(const bigfloat& x, const unsigned long y)
    {
    bigfloat a;
    mpfr_div_ui(a.x, x.x, y, mpfr_get_default_rounding_mode());
    return a;
    }

  friend bigfloat sqrt_bf(const bigfloat& x)
    {
    bigfloat a;
    mpfr_sqrt(a.x, x.x, mpfr_get_default_rounding_mode());
    return a;
    }

  friend bigfloat sqrt_bf(const unsigned long x)
    {
    bigfloat a;
    mpfr_sqrt_ui(a.x, x, mpfr_get_default_rounding_mode());
    return a;
    }

  friend bigfloat abs_bf(const bigfloat& x)
    {
    bigfloat a;
    mpfr_abs(a.x, x.x, mpfr_get_default_rounding_mode());
    return a;
    }

  friend bigfloat pow_bf(const bigfloat& a, long power) 
    {
    bigfloat b;
    mpfr_pow_ui(b.x, a.x, power, mpfr_get_default_rounding_mode());
    return b;
    }

  friend bigfloat pow_bf(const bigfloat& a, bigfloat &power) 
    {
    bigfloat b;
    mpfr_pow(b.x,a.x,power.x,GMP_RNDN);
    return b;
    }

  friend bigfloat exp_bf(const bigfloat& a) 
    {
    bigfloat b;
    mpfr_exp(b.x,a.x,GMP_RNDN);
    return b;
    }

  /* Comparison Functions */

  friend int operator>(const bigfloat& x, const bigfloat& y) 
    {
    int test;
    test = mpfr_cmp(x.x, y.x);
    if (test > 0) return 1;
    else return 0;
    }

  friend int operator<(const bigfloat& x, const bigfloat& y) 
    {
    int test;
    test = mpfr_cmp(x.x,y.x);
    if (test < 0) return 1;
    else return 0;
    }

};


#endif
