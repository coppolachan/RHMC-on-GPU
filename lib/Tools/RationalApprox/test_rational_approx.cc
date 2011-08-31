// same as rationalapprox_calc.cc
// to be used to find the correct approximations before running the rhmc program

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include"../Remez/alg_remez.h"
#include"../Remez/alg_remez.cc"

#include"../Include/global_macro.cc"
#include"../Include/parameters.cc"


int main(void)
  {
//  FILE *output = fopen("REMEZ_approx.dat", "a");

  // CALCULATION OF COEFFICIENTS FOR FIRST_INV_APPROX_NORM_COEFF

  int n=approx_metro; // The degree of the numerator polynomial
  int d=approx_metro; // The degree of the denominator polynomial
  int y1=no_flavours; // The numerator of the exponent
  int z1=8*no_ps;     // The denominator of the exponent
  int precision=gmp_remez_precision; // The precision that gmp uses
  double lambda_low=lambda_min_metro;
  double  lambda_high=1.0;              // The bounds of the approximation

  // The error from the approximation (the relative error is minimised
  // - if another error minimisation is requried, then line 398 in
  // alg_remez.C is where to change it)
  double error;

  // The partial fraction expansion takes the form 
  // r(x) = norm + sum_{k=1}^{n} res[k] / (x + pole[k])
  double norm;
  double *res1 = new double[n];
  double *pole1 = new double[d];
 
  printf("\nApproximation to f(x) = (x)^(%d/%d) on [%e, %e]\n\n", y1, z1, lambda_low, lambda_high);
  fflush(stdout);

  // Instantiate the Remez class
  AlgRemez remez1(lambda_low,lambda_high,precision);

  // Generate the required approximation
  error = remez1.generateApprox(n,d,y1,z1);

  printf("approximation error = %e\n\n", error);

  // Find the partial fraction expansion of the approximation 
  // to the function x^{y/z} (this only works currently for 
  // the special case that n = d)
  remez1.getPFE(res1,pole1,&norm);

/*
  fprintf(output, "\nApproximation to f(x) = (x)^(%d/%d)\n\n", y1, z1);
  fprintf(output, "RA_a0 = %18.16e\n", norm);
  for(int i = 0; i < n; i++) 
     {
     fprintf(output, "RA_a[%d] = %18.16e, RA_b[%d] = %18.16e\n", i, res1[i], i, pole1[i]);
     } 
*/

  printf("RA_a0 = %18.16e\n", norm);
  for(int i = 0; i < n; i++) 
     {
     printf("RA_a[%d] = %18.16e, RA_b[%d] = %18.16e\n", i, res1[i], i, pole1[i]);
     } 

//  first_inv_approx_norm_coeff= new RationalApprox(lambda_low, n, norm, res1, pole1);  

  delete res1;
  delete pole1;



  // CALCULATION OF COEFFICIENTS FOR MD_INV_APPROX_NORM_COEFF

  n=approx_md; // The degree of the numerator polynomial
  d=approx_md; // The degree of the denominator polynomial
  y1=no_flavours; // The numerator of the exponent
  z1=4*no_ps;     // The denominator of the exponent
  precision=gmp_remez_precision; // The precision that gmp uses
  lambda_low=lambda_min_md;      // The lower bounds of the approximation

  double *res2 = new double[n];
  double *pole2 = new double[d];

  printf("\nApproximation to f(x) = (x)^(-%d/%d) on [%e, %e]\n\n", y1, z1, lambda_low, lambda_high);
  fflush(stdout);
  // Instantiate the Remez class
  AlgRemez remez2(lambda_low,lambda_high,precision);

  // Generate the required approximation
  error = remez2.generateApprox(n,d,y1,z1);

  printf("approximation error = %e\n\n", error);

  // Find pfe of inverse function
  remez2.getIPFE(res2,pole2,&norm);

/*
  fprintf(output, "\nApproximation to f(x) = (x)^(-%d/%d)\n\n", y1, z1);
  fprintf(output, "RA_a0 = %18.16e\n", norm);
  for(int i = 0; i < n; i++) 
     {
     fprintf(output, "RA_a[%d] = %18.16e, RA_b[%d] = %18.16e\n", i, res2[i], i, pole2[i]);
     } 
*/

  printf("RA_a0 = %18.16e\n", norm);
  for(int i = 0; i < n; i++) 
     {
     printf("RA_a[%d] = %18.16e, RA_b[%d] = %18.16e\n", i, res2[i], i, pole2[i]);
     } 

//  md_inv_approx_norm_coeff=new RationalApprox(lambda_low, n, norm, res2, pole2);  

  delete res2;
  delete pole2;


  // CALCULATION OF COEFFICIENTS FOR LAST_INV_APPROX_NORM_COEFF

  n=approx_metro; // The degree of the numerator polynomial
  d=approx_metro; // The degree of the denominator polynomial
  y1=no_flavours; // The numerator of the exponent
  z1=4*no_ps;     // The denominator of the exponent
  precision=gmp_remez_precision; // The precision that gmp uses
  lambda_low=lambda_min_metro;      // The lower bounds of the approximation

  double *res3 = new double[n];
  double *pole3 = new double[d];

  printf("\nApproximation to f(x) = (x)^(-%d/%d) on [%e, %e]\n\n", y1, z1, lambda_low, lambda_high);
  fflush(stdout);

  // Instantiate the Remez class
  AlgRemez remez3(lambda_low,lambda_high,precision);

  // Generate the required approximation
  error = remez3.generateApprox(n,d,y1,z1);

  printf("approximation error = %e\n\n", error);

  // Find pfe of inverse function
  remez3.getIPFE(res3,pole3,&norm);

/*
  fprintf(output, "\nApproximation to f(x) = (x)^(-%d/%d)\n\n", y1, z1);
  fprintf(output, "RA_a0 = %18.16e\n", norm);
  for(int i = 0; i < n; i++) 
     {
     fprintf(output, "RA_a[%d] = %18.16e, RA_b[%d] = %18.16e\n", i, res3[i], i, pole3[i]);
     } 
*/

  printf("RA_a0 = %18.16e\n", norm);
  for(int i = 0; i < n; i++) 
     {
     printf("RA_a[%d] = %18.16e, RA_b[%d] = %18.16e\n", i, res3[i], i, pole3[i]);
     } 

//  last_inv_approx_norm_coeff= new RationalApprox(lambda_low, n, norm, res3, pole3);  


//  fclose(output);

  delete res3;
  delete pole3;
  }
