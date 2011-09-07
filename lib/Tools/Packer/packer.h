#ifndef PACKER_H
#define PACKER_H

#include "include/global_var.h"

void smartpack_gauge(float out[2*12*no_links] , const Conf *in);
void smartunpack_gauge(Conf *out, const float in[2*12*no_links]);

void smartpack_fermion(float out[6*sizeh], const Fermion *in);
void smartpack_fermion_d(float out[6*sizeh*2], const Fermion *in);
void smartunpack_fermion_d(Fermion *out, const float in[6*sizeh*2]);

void smartpack_multifermion(float *out, const MultiFermion *in);
void smartunpack_multifermion(MultiFermion *out, const float *in);

void smartunpack_multishiftfermion(ShiftMultiFermion *out, const float *in,  int order);

void make_shift_table(int table[8*size]);

void smartpack_tamatrix(float out[8*no_links], Ipdot *in);
void smartunpack_tamatrix(Ipdot *out, float in[8*no_links]);

void smartpack_thmatrix(float out[8*no_links], Momenta *in);
void smartunpack_thmatrix(Momenta *out, float in[8*no_links]);

#endif
