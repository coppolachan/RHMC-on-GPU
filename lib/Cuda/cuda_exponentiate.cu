
// in1, in2 <-- momenta = traceless hermitian
// A_{ab}=i*momenta = traceless antihermitian
#define A00_RE 0
#define A00_IM in2.z 
#define A01_RE (-in1.y)
#define A01_IM in1.x
#define A02_RE (-in1.w)
#define A02_IM in1.z

#define A10_RE in1.y
#define A10_IM in1.x
#define A11_RE 0
#define A11_IM in2.w
#define A12_RE (-in2.y)
#define A12_IM in2.x

#define A20_RE in1.w
#define A20_IM in1.z
#define A21_RE in2.y
#define A21_IM in2.x
#define A22_RE 0
#define A22_IM (-in2.z -in2.w)



__global__ void ExponentiateKernel(float4 *mom,
				   float2 *expon)
  {
  //Linear index
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  float4 in1, in2;

  float2 ris_00, ris_01, ris_02,
         ris_10, ris_11, ris_12,
         ris_20, ris_21, ris_22;

  float2 aux0, aux1, aux2;

  in1 = mom[idx + size_dev*(0+2*blockIdx.y)];	
  in2 = mom[idx + size_dev*(1+2*blockIdx.y)];	

  in1.x = in1.x*f_aux_dev;
  in1.y = in1.y*f_aux_dev;
  in1.z = in1.z*f_aux_dev;
  in1.w = in1.w*f_aux_dev;
  
  in2.x = in2.x*f_aux_dev;
  in2.y = in2.y*f_aux_dev;
  in2.z = in2.z*f_aux_dev;
  in2.w = in2.w*f_aux_dev;

  // exp x = 1+x/2*(1+x/3*(1+x/4*(1+x/5)))

  // first iteration
  // ris=1+x/5
  ris_00.x=1.0f+A00_RE*0.2f;
  ris_00.y=A00_IM*0.2f;
  ris_01.x=A01_RE*0.2f;
  ris_01.y=A01_IM*0.2f;
  ris_02.x=A02_RE*0.2f;
  ris_02.y=A02_IM*0.2f;

  ris_10.x=A10_RE*0.2f;
  ris_10.y=A10_IM*0.2f;
  ris_11.x=1.0f+A11_RE*0.2f;
  ris_11.y=A11_IM*0.2f;
  ris_12.x=A12_RE*0.2f;
  ris_12.y=A12_IM*0.2f;

  ris_20.x=A20_RE*0.2f;
  ris_20.y=A20_IM*0.2f;
  ris_21.x=A21_RE*0.2f;
  ris_21.y=A21_IM*0.2f;
  ris_22.x=1.0f+A22_RE*0.2f;
  ris_22.y=A22_IM*0.2f;

  // second iteration
  // ris = 1+x/4*(1+x/5)
  aux0.x=1.0f+ris_00.x*A00_RE*0.25f-ris_00.y*A00_IM*0.25f
             +ris_01.x*A10_RE*0.25f-ris_01.y*A10_IM*0.25f
             +ris_02.x*A20_RE*0.25f-ris_02.y*A20_IM*0.25f;
  aux0.y=    +ris_00.y*A00_RE*0.25f+ris_00.x*A00_IM*0.25f
             +ris_01.y*A10_RE*0.25f+ris_01.x*A10_IM*0.25f
             +ris_02.y*A20_RE*0.25f+ris_02.x*A20_IM*0.25f;

  aux1.x=    +ris_00.x*A01_RE*0.25f-ris_00.y*A01_IM*0.25f
             +ris_01.x*A11_RE*0.25f-ris_01.y*A11_IM*0.25f
             +ris_02.x*A21_RE*0.25f-ris_02.y*A21_IM*0.25f;
  aux1.y=    +ris_00.y*A01_RE*0.25f+ris_00.x*A01_IM*0.25f
             +ris_01.y*A11_RE*0.25f+ris_01.x*A11_IM*0.25f
             +ris_02.y*A21_RE*0.25f+ris_02.x*A21_IM*0.25f;
 
  aux2.x=    +ris_00.x*A02_RE*0.25f-ris_00.y*A02_IM*0.25f
             +ris_01.x*A12_RE*0.25f-ris_01.y*A12_IM*0.25f
             +ris_02.x*A22_RE*0.25f-ris_02.y*A22_IM*0.25f;
  aux2.y=    +ris_00.y*A02_RE*0.25f+ris_00.x*A02_IM*0.25f
             +ris_01.y*A12_RE*0.25f+ris_01.x*A12_IM*0.25f
             +ris_02.y*A22_RE*0.25f+ris_02.x*A22_IM*0.25f;

  ris_00=aux0;
  ris_01=aux1;
  ris_02=aux2;

  aux0.x=    +ris_10.x*A00_RE*0.25f-ris_10.y*A00_IM*0.25f
             +ris_11.x*A10_RE*0.25f-ris_11.y*A10_IM*0.25f
             +ris_12.x*A20_RE*0.25f-ris_12.y*A20_IM*0.25f;
  aux0.y=    +ris_10.y*A00_RE*0.25f+ris_10.x*A00_IM*0.25f
             +ris_11.y*A10_RE*0.25f+ris_11.x*A10_IM*0.25f
             +ris_12.y*A20_RE*0.25f+ris_12.x*A20_IM*0.25f;

  aux1.x=1.0f+ris_10.x*A01_RE*0.25f-ris_10.y*A01_IM*0.25f
             +ris_11.x*A11_RE*0.25f-ris_11.y*A11_IM*0.25f
             +ris_12.x*A21_RE*0.25f-ris_12.y*A21_IM*0.25f;
  aux1.y=    +ris_10.y*A01_RE*0.25f+ris_10.x*A01_IM*0.25f
             +ris_11.y*A11_RE*0.25f+ris_11.x*A11_IM*0.25f
             +ris_12.y*A21_RE*0.25f+ris_12.x*A21_IM*0.25f;
 
  aux2.x=    +ris_10.x*A02_RE*0.25f-ris_10.y*A02_IM*0.25f
             +ris_11.x*A12_RE*0.25f-ris_11.y*A12_IM*0.25f
             +ris_12.x*A22_RE*0.25f-ris_12.y*A22_IM*0.25f;
  aux2.y=    +ris_10.y*A02_RE*0.25f+ris_10.x*A02_IM*0.25f
             +ris_11.y*A12_RE*0.25f+ris_11.x*A12_IM*0.25f
             +ris_12.y*A22_RE*0.25f+ris_12.x*A22_IM*0.25f;

  ris_10=aux0;
  ris_11=aux1;
  ris_12=aux2;

  aux0.x=    +ris_20.x*A00_RE*0.25f-ris_20.y*A00_IM*0.25f
             +ris_21.x*A10_RE*0.25f-ris_21.y*A10_IM*0.25f
             +ris_22.x*A20_RE*0.25f-ris_22.y*A20_IM*0.25f;
  aux0.y=    +ris_20.y*A00_RE*0.25f+ris_20.x*A00_IM*0.25f
             +ris_21.y*A10_RE*0.25f+ris_21.x*A10_IM*0.25f
             +ris_22.y*A20_RE*0.25f+ris_22.x*A20_IM*0.25f;

  aux1.x=    +ris_20.x*A01_RE*0.25f-ris_20.y*A01_IM*0.25f
             +ris_21.x*A11_RE*0.25f-ris_21.y*A11_IM*0.25f
             +ris_22.x*A21_RE*0.25f-ris_22.y*A21_IM*0.25f;
  aux1.y=    +ris_20.y*A01_RE*0.25f+ris_20.x*A01_IM*0.25f
             +ris_21.y*A11_RE*0.25f+ris_21.x*A11_IM*0.25f
             +ris_22.y*A21_RE*0.25f+ris_22.x*A21_IM*0.25f;
 
  aux2.x=1.0f+ris_20.x*A02_RE*0.25f-ris_20.y*A02_IM*0.25f
             +ris_21.x*A12_RE*0.25f-ris_21.y*A12_IM*0.25f
             +ris_22.x*A22_RE*0.25f-ris_22.y*A22_IM*0.25f;
  aux2.y=    +ris_20.y*A02_RE*0.25f+ris_20.x*A02_IM*0.25f
             +ris_21.y*A12_RE*0.25f+ris_21.x*A12_IM*0.25f
             +ris_22.y*A22_RE*0.25f+ris_22.x*A22_IM*0.25f;

  ris_20=aux0;
  ris_21=aux1;
  ris_22=aux2;

  // third iteration
  // ris = 1+x/3*(1+x/4*(1+x/5))
  aux0.x=1.0f+ris_00.x*A00_RE*0.3333333f-ris_00.y*A00_IM*0.3333333f
             +ris_01.x*A10_RE*0.3333333f-ris_01.y*A10_IM*0.3333333f
             +ris_02.x*A20_RE*0.3333333f-ris_02.y*A20_IM*0.3333333f;
  aux0.y=    +ris_00.y*A00_RE*0.3333333f+ris_00.x*A00_IM*0.3333333f
             +ris_01.y*A10_RE*0.3333333f+ris_01.x*A10_IM*0.3333333f
             +ris_02.y*A20_RE*0.3333333f+ris_02.x*A20_IM*0.3333333f;

  aux1.x=    +ris_00.x*A01_RE*0.3333333f-ris_00.y*A01_IM*0.3333333f
             +ris_01.x*A11_RE*0.3333333f-ris_01.y*A11_IM*0.3333333f
             +ris_02.x*A21_RE*0.3333333f-ris_02.y*A21_IM*0.3333333f;
  aux1.y=    +ris_00.y*A01_RE*0.3333333f+ris_00.x*A01_IM*0.3333333f
             +ris_01.y*A11_RE*0.3333333f+ris_01.x*A11_IM*0.3333333f
             +ris_02.y*A21_RE*0.3333333f+ris_02.x*A21_IM*0.3333333f;
 
  aux2.x=    +ris_00.x*A02_RE*0.3333333f-ris_00.y*A02_IM*0.3333333f
             +ris_01.x*A12_RE*0.3333333f-ris_01.y*A12_IM*0.3333333f
             +ris_02.x*A22_RE*0.3333333f-ris_02.y*A22_IM*0.3333333f;
  aux2.y=    +ris_00.y*A02_RE*0.3333333f+ris_00.x*A02_IM*0.3333333f
             +ris_01.y*A12_RE*0.3333333f+ris_01.x*A12_IM*0.3333333f
             +ris_02.y*A22_RE*0.3333333f+ris_02.x*A22_IM*0.3333333f;

  ris_00=aux0;
  ris_01=aux1;
  ris_02=aux2;

  aux0.x=    +ris_10.x*A00_RE*0.3333333f-ris_10.y*A00_IM*0.3333333f
             +ris_11.x*A10_RE*0.3333333f-ris_11.y*A10_IM*0.3333333f
             +ris_12.x*A20_RE*0.3333333f-ris_12.y*A20_IM*0.3333333f;
  aux0.y=    +ris_10.y*A00_RE*0.3333333f+ris_10.x*A00_IM*0.3333333f
             +ris_11.y*A10_RE*0.3333333f+ris_11.x*A10_IM*0.3333333f
             +ris_12.y*A20_RE*0.3333333f+ris_12.x*A20_IM*0.3333333f;

  aux1.x=1.0f+ris_10.x*A01_RE*0.3333333f-ris_10.y*A01_IM*0.3333333f
             +ris_11.x*A11_RE*0.3333333f-ris_11.y*A11_IM*0.3333333f
             +ris_12.x*A21_RE*0.3333333f-ris_12.y*A21_IM*0.3333333f;
  aux1.y=    +ris_10.y*A01_RE*0.3333333f+ris_10.x*A01_IM*0.3333333f
             +ris_11.y*A11_RE*0.3333333f+ris_11.x*A11_IM*0.3333333f
             +ris_12.y*A21_RE*0.3333333f+ris_12.x*A21_IM*0.3333333f;
 
  aux2.x=    +ris_10.x*A02_RE*0.3333333f-ris_10.y*A02_IM*0.3333333f
             +ris_11.x*A12_RE*0.3333333f-ris_11.y*A12_IM*0.3333333f
             +ris_12.x*A22_RE*0.3333333f-ris_12.y*A22_IM*0.3333333f;
  aux2.y=    +ris_10.y*A02_RE*0.3333333f+ris_10.x*A02_IM*0.3333333f
             +ris_11.y*A12_RE*0.3333333f+ris_11.x*A12_IM*0.3333333f
             +ris_12.y*A22_RE*0.3333333f+ris_12.x*A22_IM*0.3333333f;

  ris_10=aux0;
  ris_11=aux1;
  ris_12=aux2;

  aux0.x=    +ris_20.x*A00_RE*0.3333333f-ris_20.y*A00_IM*0.3333333f
             +ris_21.x*A10_RE*0.3333333f-ris_21.y*A10_IM*0.3333333f
             +ris_22.x*A20_RE*0.3333333f-ris_22.y*A20_IM*0.3333333f;
  aux0.y=    +ris_20.y*A00_RE*0.3333333f+ris_20.x*A00_IM*0.3333333f
             +ris_21.y*A10_RE*0.3333333f+ris_21.x*A10_IM*0.3333333f
             +ris_22.y*A20_RE*0.3333333f+ris_22.x*A20_IM*0.3333333f;

  aux1.x=    +ris_20.x*A01_RE*0.3333333f-ris_20.y*A01_IM*0.3333333f
             +ris_21.x*A11_RE*0.3333333f-ris_21.y*A11_IM*0.3333333f
             +ris_22.x*A21_RE*0.3333333f-ris_22.y*A21_IM*0.3333333f;
  aux1.y=    +ris_20.y*A01_RE*0.3333333f+ris_20.x*A01_IM*0.3333333f
             +ris_21.y*A11_RE*0.3333333f+ris_21.x*A11_IM*0.3333333f
             +ris_22.y*A21_RE*0.3333333f+ris_22.x*A21_IM*0.3333333f;
 
  aux2.x=1.0f+ris_20.x*A02_RE*0.3333333f-ris_20.y*A02_IM*0.3333333f
             +ris_21.x*A12_RE*0.3333333f-ris_21.y*A12_IM*0.3333333f
             +ris_22.x*A22_RE*0.3333333f-ris_22.y*A22_IM*0.3333333f;
  aux2.y=    +ris_20.y*A02_RE*0.3333333f+ris_20.x*A02_IM*0.3333333f
             +ris_21.y*A12_RE*0.3333333f+ris_21.x*A12_IM*0.3333333f
             +ris_22.y*A22_RE*0.3333333f+ris_22.x*A22_IM*0.3333333f;

  ris_20=aux0;
  ris_21=aux1;
  ris_22=aux2;

  // 4 iteration
  // ris = 1+x/2(1+x/3*(1+x/4*(1+x/5)))
  aux0.x=1.0f+ris_00.x*A00_RE*0.5f-ris_00.y*A00_IM*0.5f
             +ris_01.x*A10_RE*0.5f-ris_01.y*A10_IM*0.5f
             +ris_02.x*A20_RE*0.5f-ris_02.y*A20_IM*0.5f;
  aux0.y=    +ris_00.y*A00_RE*0.5f+ris_00.x*A00_IM*0.5f
             +ris_01.y*A10_RE*0.5f+ris_01.x*A10_IM*0.5f
             +ris_02.y*A20_RE*0.5f+ris_02.x*A20_IM*0.5f;

  aux1.x=    +ris_00.x*A01_RE*0.5f-ris_00.y*A01_IM*0.5f
             +ris_01.x*A11_RE*0.5f-ris_01.y*A11_IM*0.5f
             +ris_02.x*A21_RE*0.5f-ris_02.y*A21_IM*0.5f;
  aux1.y=    +ris_00.y*A01_RE*0.5f+ris_00.x*A01_IM*0.5f
             +ris_01.y*A11_RE*0.5f+ris_01.x*A11_IM*0.5f
             +ris_02.y*A21_RE*0.5f+ris_02.x*A21_IM*0.5f;
 
  aux2.x=    +ris_00.x*A02_RE*0.5f-ris_00.y*A02_IM*0.5f
             +ris_01.x*A12_RE*0.5f-ris_01.y*A12_IM*0.5f
             +ris_02.x*A22_RE*0.5f-ris_02.y*A22_IM*0.5f;
  aux2.y=    +ris_00.y*A02_RE*0.5f+ris_00.x*A02_IM*0.5f
             +ris_01.y*A12_RE*0.5f+ris_01.x*A12_IM*0.5f
             +ris_02.y*A22_RE*0.5f+ris_02.x*A22_IM*0.5f;

  ris_00=aux0;
  ris_01=aux1;
  ris_02=aux2;

  aux0.x=    +ris_10.x*A00_RE*0.5f-ris_10.y*A00_IM*0.5f
             +ris_11.x*A10_RE*0.5f-ris_11.y*A10_IM*0.5f
             +ris_12.x*A20_RE*0.5f-ris_12.y*A20_IM*0.5f;
  aux0.y=    +ris_10.y*A00_RE*0.5f+ris_10.x*A00_IM*0.5f
             +ris_11.y*A10_RE*0.5f+ris_11.x*A10_IM*0.5f
             +ris_12.y*A20_RE*0.5f+ris_12.x*A20_IM*0.5f;

  aux1.x=1.0f+ris_10.x*A01_RE*0.5f-ris_10.y*A01_IM*0.5f
             +ris_11.x*A11_RE*0.5f-ris_11.y*A11_IM*0.5f
             +ris_12.x*A21_RE*0.5f-ris_12.y*A21_IM*0.5f;
  aux1.y=    +ris_10.y*A01_RE*0.5f+ris_10.x*A01_IM*0.5f
             +ris_11.y*A11_RE*0.5f+ris_11.x*A11_IM*0.5f
             +ris_12.y*A21_RE*0.5f+ris_12.x*A21_IM*0.5f;
 
  aux2.x=    +ris_10.x*A02_RE*0.5f-ris_10.y*A02_IM*0.5f
             +ris_11.x*A12_RE*0.5f-ris_11.y*A12_IM*0.5f
             +ris_12.x*A22_RE*0.5f-ris_12.y*A22_IM*0.5f;
  aux2.y=    +ris_10.y*A02_RE*0.5f+ris_10.x*A02_IM*0.5f
             +ris_11.y*A12_RE*0.5f+ris_11.x*A12_IM*0.5f
             +ris_12.y*A22_RE*0.5f+ris_12.x*A22_IM*0.5f;

  ris_10=aux0;
  ris_11=aux1;
  ris_12=aux2;

  aux0.x=    +ris_20.x*A00_RE*0.5f-ris_20.y*A00_IM*0.5f
             +ris_21.x*A10_RE*0.5f-ris_21.y*A10_IM*0.5f
             +ris_22.x*A20_RE*0.5f-ris_22.y*A20_IM*0.5f;
  aux0.y=    +ris_20.y*A00_RE*0.5f+ris_20.x*A00_IM*0.5f
             +ris_21.y*A10_RE*0.5f+ris_21.x*A10_IM*0.5f
             +ris_22.y*A20_RE*0.5f+ris_22.x*A20_IM*0.5f;

  aux1.x=    +ris_20.x*A01_RE*0.5f-ris_20.y*A01_IM*0.5f
             +ris_21.x*A11_RE*0.5f-ris_21.y*A11_IM*0.5f
             +ris_22.x*A21_RE*0.5f-ris_22.y*A21_IM*0.5f;
  aux1.y=    +ris_20.y*A01_RE*0.5f+ris_20.x*A01_IM*0.5f
             +ris_21.y*A11_RE*0.5f+ris_21.x*A11_IM*0.5f
             +ris_22.y*A21_RE*0.5f+ris_22.x*A21_IM*0.5f;
 
  aux2.x=1.0f+ris_20.x*A02_RE*0.5f-ris_20.y*A02_IM*0.5f
             +ris_21.x*A12_RE*0.5f-ris_21.y*A12_IM*0.5f
             +ris_22.x*A22_RE*0.5f-ris_22.y*A22_IM*0.5f;
  aux2.y=    +ris_20.y*A02_RE*0.5f+ris_20.x*A02_IM*0.5f
             +ris_21.y*A12_RE*0.5f+ris_21.x*A12_IM*0.5f
             +ris_22.y*A22_RE*0.5f+ris_22.x*A22_IM*0.5f;

  ris_20=aux0;
  ris_21=aux1;
  ris_22=aux2;

  // 5 iteration
  // ris = 1+x*(1+x/2(1+x/3*(1+x/4*(1+x/5))))
  aux0.x=1.0f+ris_00.x*A00_RE-ris_00.y*A00_IM
             +ris_01.x*A10_RE-ris_01.y*A10_IM
             +ris_02.x*A20_RE-ris_02.y*A20_IM;
  aux0.y=    +ris_00.y*A00_RE+ris_00.x*A00_IM
             +ris_01.y*A10_RE+ris_01.x*A10_IM
             +ris_02.y*A20_RE+ris_02.x*A20_IM;

  aux1.x=    +ris_00.x*A01_RE-ris_00.y*A01_IM
             +ris_01.x*A11_RE-ris_01.y*A11_IM
             +ris_02.x*A21_RE-ris_02.y*A21_IM;
  aux1.y=    +ris_00.y*A01_RE+ris_00.x*A01_IM
             +ris_01.y*A11_RE+ris_01.x*A11_IM
             +ris_02.y*A21_RE+ris_02.x*A21_IM;
 
  aux2.x=    +ris_00.x*A02_RE-ris_00.y*A02_IM
             +ris_01.x*A12_RE-ris_01.y*A12_IM
             +ris_02.x*A22_RE-ris_02.y*A22_IM;
  aux2.y=    +ris_00.y*A02_RE+ris_00.x*A02_IM
             +ris_01.y*A12_RE+ris_01.x*A12_IM
             +ris_02.y*A22_RE+ris_02.x*A22_IM;

  ris_00=aux0;
  ris_01=aux1;
  ris_02=aux2;

  aux0.x=    +ris_10.x*A00_RE-ris_10.y*A00_IM
             +ris_11.x*A10_RE-ris_11.y*A10_IM
             +ris_12.x*A20_RE-ris_12.y*A20_IM;
  aux0.y=    +ris_10.y*A00_RE+ris_10.x*A00_IM
             +ris_11.y*A10_RE+ris_11.x*A10_IM
             +ris_12.y*A20_RE+ris_12.x*A20_IM;

  aux1.x=1.0f+ris_10.x*A01_RE-ris_10.y*A01_IM
             +ris_11.x*A11_RE-ris_11.y*A11_IM
             +ris_12.x*A21_RE-ris_12.y*A21_IM;
  aux1.y=    +ris_10.y*A01_RE+ris_10.x*A01_IM
             +ris_11.y*A11_RE+ris_11.x*A11_IM
             +ris_12.y*A21_RE+ris_12.x*A21_IM;
 
  aux2.x=    +ris_10.x*A02_RE-ris_10.y*A02_IM
             +ris_11.x*A12_RE-ris_11.y*A12_IM
             +ris_12.x*A22_RE-ris_12.y*A22_IM;
  aux2.y=    +ris_10.y*A02_RE+ris_10.x*A02_IM
             +ris_11.y*A12_RE+ris_11.x*A12_IM
             +ris_12.y*A22_RE+ris_12.x*A22_IM;

  ris_10=aux0;
  ris_11=aux1;
  ris_12=aux2;

  aux0.x=    +ris_20.x*A00_RE-ris_20.y*A00_IM
             +ris_21.x*A10_RE-ris_21.y*A10_IM
             +ris_22.x*A20_RE-ris_22.y*A20_IM;
  aux0.y=    +ris_20.y*A00_RE+ris_20.x*A00_IM
             +ris_21.y*A10_RE+ris_21.x*A10_IM
             +ris_22.y*A20_RE+ris_22.x*A20_IM;

  aux1.x=    +ris_20.x*A01_RE-ris_20.y*A01_IM
             +ris_21.x*A11_RE-ris_21.y*A11_IM
             +ris_22.x*A21_RE-ris_22.y*A21_IM;
  aux1.y=    +ris_20.y*A01_RE+ris_20.x*A01_IM
             +ris_21.y*A11_RE+ris_21.x*A11_IM
             +ris_22.y*A21_RE+ris_22.x*A21_IM;
 
  aux2.x=1.0f+ris_20.x*A02_RE-ris_20.y*A02_IM
             +ris_21.x*A12_RE-ris_21.y*A12_IM
             +ris_22.x*A22_RE-ris_22.y*A22_IM;
  aux2.y=    +ris_20.y*A02_RE+ris_20.x*A02_IM
             +ris_21.y*A12_RE+ris_21.x*A12_IM
             +ris_22.y*A22_RE+ris_22.x*A22_IM;

  ris_20=aux0;
  ris_21=aux1;
  ris_22=aux2;

  expon[idx+            +9*size_dev*blockIdx.y]=ris_00;
  expon[idx+ 1*size_dev +9*size_dev*blockIdx.y]=ris_01;
  expon[idx+ 2*size_dev +9*size_dev*blockIdx.y]=ris_02;

  expon[idx+ 3*size_dev +9*size_dev*blockIdx.y]=ris_10;
  expon[idx+ 4*size_dev +9*size_dev*blockIdx.y]=ris_11;
  expon[idx+ 5*size_dev +9*size_dev*blockIdx.y]=ris_12;

  expon[idx+ 6*size_dev +9*size_dev*blockIdx.y]=ris_20;
  expon[idx+ 7*size_dev +9*size_dev*blockIdx.y]=ris_21;
  expon[idx+ 8*size_dev +9*size_dev*blockIdx.y]=ris_22;
  }




__global__ void MultiplyByFieldKernel(float2 *temporary, 
                                      const int *phases, 
                                      size_t gauge_offset)
  {
  // temporary = temporary * gauge_field  
  // input gauge_field is a SU(3) matrix, temporary is a generic 3x3 matrix
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int stag_phase=phases[idx+blockIdx.y*size_dev];

  float4 link0, link1, link2;

  //Temporary row-column store
  float2 mat0, mat1, mat2;  //6 registers
  float2 row[3];

  LoadLinkRegs(gauge_texRef, size_dev, idx + gauge_offset, blockIdx.y);

  row[0]=temporary[idx+            +9*size_dev*blockIdx.y];
  row[1]=temporary[idx+ 1*size_dev +9*size_dev*blockIdx.y];
  row[2]=temporary[idx+ 2*size_dev +9*size_dev*blockIdx.y];

  mat0.x =             row[0].x*link0.x - row[0].y*link0.y +
                       row[1].x*link1.z - row[1].y*link1.w +
           stag_phase*(row[2].x*C1RE    - row[2].y*C1IM);

  mat0.y =             row[0].x*link0.y + row[0].y*link0.x +
                       row[1].x*link1.w + row[1].y*link1.z +
           stag_phase*(row[2].x*C1IM    + row[2].y*C1RE);

  mat1.x =             row[0].x*link0.z - row[0].y*link0.w +
                       row[1].x*link2.x - row[1].y*link2.y +
           stag_phase*(row[2].x*C2RE    - row[2].y*C2IM);

  mat1.y =             row[0].x*link0.w + row[0].y*link0.z +
                       row[1].x*link2.y + row[1].y*link2.x +
           stag_phase*(row[2].x*C2IM    + row[2].y*C2RE);

  mat2.x =             row[0].x*link1.x - row[0].y*link1.y +
                       row[1].x*link2.z - row[1].y*link2.w +
           stag_phase*(row[2].x*C3RE    - row[2].y*C3IM);

  mat2.y =             row[0].x*link1.y + row[0].y*link1.x +
                       row[1].x*link2.w + row[1].y*link2.z +
           stag_phase*(row[2].x*C3IM    + row[2].y*C3RE);

  temporary[idx+            +9*size_dev*blockIdx.y]=mat0;
  temporary[idx+ 1*size_dev +9*size_dev*blockIdx.y]=mat1;
  temporary[idx+ 2*size_dev +9*size_dev*blockIdx.y]=mat2;


  row[0]=temporary[idx+ 3*size_dev +9*size_dev*blockIdx.y];
  row[1]=temporary[idx+ 4*size_dev +9*size_dev*blockIdx.y];
  row[2]=temporary[idx+ 5*size_dev +9*size_dev*blockIdx.y];

  mat0.x =             row[0].x*link0.x - row[0].y*link0.y +
                       row[1].x*link1.z - row[1].y*link1.w +
           stag_phase*(row[2].x*C1RE    - row[2].y*C1IM);

  mat0.y =             row[0].x*link0.y + row[0].y*link0.x +
                       row[1].x*link1.w + row[1].y*link1.z +
           stag_phase*(row[2].x*C1IM    + row[2].y*C1RE);

  mat1.x =             row[0].x*link0.z - row[0].y*link0.w +
                       row[1].x*link2.x - row[1].y*link2.y +
           stag_phase*(row[2].x*C2RE    - row[2].y*C2IM);

  mat1.y =             row[0].x*link0.w + row[0].y*link0.z +
                       row[1].x*link2.y + row[1].y*link2.x +
           stag_phase*(row[2].x*C2IM    + row[2].y*C2RE);

  mat2.x =             row[0].x*link1.x - row[0].y*link1.y +
                       row[1].x*link2.z - row[1].y*link2.w +
           stag_phase*(row[2].x*C3RE    - row[2].y*C3IM);

  mat2.y =             row[0].x*link1.y + row[0].y*link1.x +
                       row[1].x*link2.w + row[1].y*link2.z +
           stag_phase*(row[2].x*C3IM    + row[2].y*C3RE);

  temporary[idx+ 3*size_dev +9*size_dev*blockIdx.y]=mat0;
  temporary[idx+ 4*size_dev +9*size_dev*blockIdx.y]=mat1;
  temporary[idx+ 5*size_dev +9*size_dev*blockIdx.y]=mat2;


  row[0]=temporary[idx+ 6*size_dev +9*size_dev*blockIdx.y];
  row[1]=temporary[idx+ 7*size_dev +9*size_dev*blockIdx.y];
  row[2]=temporary[idx+ 8*size_dev +9*size_dev*blockIdx.y];

  mat0.x =             row[0].x*link0.x - row[0].y*link0.y +
                       row[1].x*link1.z - row[1].y*link1.w +
           stag_phase*(row[2].x*C1RE    - row[2].y*C1IM);

  mat0.y =             row[0].x*link0.y + row[0].y*link0.x +
                       row[1].x*link1.w + row[1].y*link1.z +
           stag_phase*(row[2].x*C1IM    + row[2].y*C1RE);

  mat1.x =             row[0].x*link0.z - row[0].y*link0.w +
                       row[1].x*link2.x - row[1].y*link2.y +
           stag_phase*(row[2].x*C2RE    - row[2].y*C2IM);

  mat1.y =             row[0].x*link0.w + row[0].y*link0.z +
                       row[1].x*link2.y + row[1].y*link2.x +
           stag_phase*(row[2].x*C2IM    + row[2].y*C2RE);

  mat2.x =             row[0].x*link1.x - row[0].y*link1.y +
                       row[1].x*link2.z - row[1].y*link2.w +
           stag_phase*(row[2].x*C3RE    - row[2].y*C3IM);

  mat2.y =             row[0].x*link1.y + row[0].y*link1.x +
                       row[1].x*link2.w + row[1].y*link2.z +
           stag_phase*(row[2].x*C3IM    + row[2].y*C3RE);

  temporary[idx+ 6*size_dev +9*size_dev*blockIdx.y]=mat0;
  temporary[idx+ 7*size_dev +9*size_dev*blockIdx.y]=mat1;
  temporary[idx+ 8*size_dev +9*size_dev*blockIdx.y]=mat2;
  }




// single precision
__global__ void ReunitarizeKernel(float2 *field,
                                  float4 *field_out)
  {
  //Linear index
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  float t4;
  float2 t2;

  float2 a_00, a_01, a_02,
         a_10, a_11, a_12;
  
  //Load first row
  a_00=field[idx+            +9*size_dev*blockIdx.y];
  a_01=field[idx+ 1*size_dev +9*size_dev*blockIdx.y];
  a_02=field[idx+ 2*size_dev +9*size_dev*blockIdx.y];

  // t4= first row norm squared  
  t4 = a_00.x*a_00.x+a_00.y*a_00.y +
       a_01.x*a_01.x+a_01.y*a_01.y +
       a_02.x*a_02.x+a_02.y*a_02.y;
  t4 = sqrtf(t4);

  // Normalize the first row
  t4 = 1.0f / t4;
  a_00.x *= t4;
  a_00.y *= t4;
  a_01.x *= t4;
  a_01.y *= t4;
  a_02.x *= t4;
  a_02.y *= t4;

  //Load second row
  a_10=field[idx+ 3*size_dev +9*size_dev*blockIdx.y];
  a_11=field[idx+ 4*size_dev +9*size_dev*blockIdx.y];
  a_12=field[idx+ 5*size_dev +9*size_dev*blockIdx.y];

  // Calculate the orthogonal component to the second row
  t2.x =  a_00.x*a_10.x + a_00.y*a_10.y +
          a_01.x*a_11.x + a_01.y*a_11.y +
          a_02.x*a_12.x + a_02.y*a_12.y;

  t2.y =  a_00.x*a_10.y - a_00.y*a_10.x +
          a_01.x*a_11.y - a_01.y*a_11.x +
          a_02.x*a_12.y - a_02.y*a_12.x;
  
  //Orthogonalize the second row relative to the first row
  // v = v - t2*u
  a_10.x -= t2.x*a_00.x - t2.y*a_00.y;
  a_10.y -= t2.x*a_00.y + t2.y*a_00.x;
  a_11.x -= t2.x*a_01.x - t2.y*a_01.y;
  a_11.y -= t2.x*a_01.y + t2.y*a_01.x;
  a_12.x -= t2.x*a_02.x - t2.y*a_02.y;
  a_12.y -= t2.x*a_02.y + t2.y*a_02.x;

  //Normalize the second row
  t4 = a_10.x*a_10.x+a_10.y*a_10.y +
       a_11.x*a_11.x+a_11.y*a_11.y +
       a_12.x*a_12.x+a_12.y*a_12.y;
  t4 = sqrtf(t4);

  t4 = 1.0f / t4;
  a_10.x *= t4;
  a_10.y *= t4;
  a_11.x *= t4;
  a_11.y *= t4;
  a_12.x *= t4;
  a_12.y *= t4;

  //Write out unitarized matrix
  field_out[idx +              3*size_dev*blockIdx.y].x = a_00.x;
  field_out[idx +              3*size_dev*blockIdx.y].y = a_00.y;
  field_out[idx +              3*size_dev*blockIdx.y].z = a_01.x;
  field_out[idx +              3*size_dev*blockIdx.y].w = a_01.y;

  field_out[idx +   size_dev + 3*size_dev*blockIdx.y].x = a_02.x;
  field_out[idx +   size_dev + 3*size_dev*blockIdx.y].y = a_02.y;
  field_out[idx +   size_dev + 3*size_dev*blockIdx.y].z = a_10.x;
  field_out[idx +   size_dev + 3*size_dev*blockIdx.y].w = a_10.y;

  field_out[idx + 2*size_dev + 3*size_dev*blockIdx.y].x = a_11.x;
  field_out[idx + 2*size_dev + 3*size_dev*blockIdx.y].y = a_11.y;
  field_out[idx + 2*size_dev + 3*size_dev*blockIdx.y].z = a_12.x;
  field_out[idx + 2*size_dev + 3*size_dev*blockIdx.y].w = a_12.y;
  }






// double precision
__global__ void ReunitarizeKernelD(float2 *field,
                                   float4 *field_out)
  {
  //Linear index
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  double t4;
  double2 t2;

  double2 a_00, a_01, a_02,
          a_10, a_11, a_12;
  
  float2 aux;
  
  //Load first row
  aux=field[idx+            +9*size_dev*blockIdx.y];
  a_00.x=(double)aux.x;
  a_00.y=(double)aux.y;
  aux=field[idx+ 1*size_dev +9*size_dev*blockIdx.y];
  a_01.x=(double)aux.x;
  a_01.y=(double)aux.y;
  aux=field[idx+ 2*size_dev +9*size_dev*blockIdx.y];
  a_02.x=(double)aux.x;
  a_02.y=(double)aux.y;

  t4 = a_00.x*a_00.x+a_00.y*a_00.y +
       a_01.x*a_01.x+a_01.y*a_01.y +
       a_02.x*a_02.x+a_02.y*a_02.y;
  t4 = rsqrt(t4);  // 1/sqrt

  // Normalize the first row
  a_00.x *= t4;
  a_00.y *= t4;
  a_01.x *= t4;
  a_01.y *= t4;
  a_02.x *= t4;
  a_02.y *= t4;

  //Load second row
  aux=field[idx+ 3*size_dev +9*size_dev*blockIdx.y];
  a_10.x=(double)aux.x;
  a_10.y=(double)aux.y;
  aux=field[idx+ 4*size_dev +9*size_dev*blockIdx.y];
  a_11.x=(double)aux.x;
  a_11.y=(double)aux.y;
  aux=field[idx+ 5*size_dev +9*size_dev*blockIdx.y];
  a_12.x=(double)aux.x;
  a_12.y=(double)aux.y;

  // Calculate the orthogonal component to the second row
  t2.x =  a_00.x*a_10.x + a_00.y*a_10.y +
          a_01.x*a_11.x + a_01.y*a_11.y +
          a_02.x*a_12.x + a_02.y*a_12.y;

  t2.y =  a_00.x*a_10.y - a_00.y*a_10.x +
          a_01.x*a_11.y - a_01.y*a_11.x +
          a_02.x*a_12.y - a_02.y*a_12.x;
  
  //Orthogonalize the second row relative to the first row
  // v = v - t2*u
  a_10.x -= t2.x*a_00.x - t2.y*a_00.y;
  a_10.y -= t2.x*a_00.y + t2.y*a_00.x;
  a_11.x -= t2.x*a_01.x - t2.y*a_01.y;
  a_11.y -= t2.x*a_01.y + t2.y*a_01.x;
  a_12.x -= t2.x*a_02.x - t2.y*a_02.y;
  a_12.y -= t2.x*a_02.y + t2.y*a_02.x;

  t4 = a_10.x*a_10.x+a_10.y*a_10.y +
       a_11.x*a_11.x+a_11.y*a_11.y +
       a_12.x*a_12.x+a_12.y*a_12.y;
  t4 = rsqrt(t4);  // 1/sqrt

  //Normalize the second row
  a_10.x *= t4;
  a_10.y *= t4;
  a_11.x *= t4;
  a_11.y *= t4;
  a_12.x *= t4;
  a_12.y *= t4;

  //Write out unitarized matrix

                             // 00.x component 
  aux.x=(float)a_00.x;
  t4=a_00.x-aux.x;
  aux.y=(float)t4;
  field_out[idx +              3*size_dev*blockIdx.y              ].x = aux.x;  // 1st float
  field_out[idx +              3*size_dev*blockIdx.y + 12*size_dev].x = aux.y;  // 2nd float
                             // 00.y component 
  aux.x=(float)a_00.y;
  t4=a_00.y-aux.x;
  aux.y=(float)t4;
  field_out[idx +              3*size_dev*blockIdx.y              ].y = aux.x;  // 1st float
  field_out[idx +              3*size_dev*blockIdx.y + 12*size_dev].y = aux.y;  // 2nd float
                             // 01.x component 
  aux.x=(float)a_01.x; 
  t4=a_01.x-aux.x;
  aux.y=(float)t4;
  field_out[idx +              3*size_dev*blockIdx.y              ].z = aux.x;  // 1st float
  field_out[idx +              3*size_dev*blockIdx.y + 12*size_dev].z = aux.y;  // 2nd float
                             // 01.y component 
  aux.x=(float)a_01.y;
  t4=a_01.y-aux.x;
  aux.y=(float)t4;
  field_out[idx +              3*size_dev*blockIdx.y              ].w = aux.x;  // 1st float
  field_out[idx +              3*size_dev*blockIdx.y + 12*size_dev].w = aux.y;  // 2nd float


                             // 02.x component 
  aux.x=(float)a_02.x;
  t4=a_02.x-aux.x;
  aux.y=(float)t4;
  field_out[idx +   size_dev + 3*size_dev*blockIdx.y              ].x = aux.x;
  field_out[idx +   size_dev + 3*size_dev*blockIdx.y + 12*size_dev].x = aux.y;
                             // 02.y component 
  aux.x=(float)a_02.y;
  t4=a_02.y-aux.x;
  aux.y=(float)t4;
  field_out[idx +   size_dev + 3*size_dev*blockIdx.y              ].y = aux.x;
  field_out[idx +   size_dev + 3*size_dev*blockIdx.y + 12*size_dev].y = aux.y;
                             // 10.x component 
  aux.x=(float)a_10.x;
  t4=a_10.x-aux.x;
  aux.y=(float)t4;
  field_out[idx +   size_dev + 3*size_dev*blockIdx.y              ].z = aux.x;
  field_out[idx +   size_dev + 3*size_dev*blockIdx.y + 12*size_dev].z = aux.y;
                             // 10.y component 
  aux.x=(float)a_10.y;
  t4=a_10.y-aux.x;
  aux.y=(float)t4;
  field_out[idx +   size_dev + 3*size_dev*blockIdx.y              ].w = aux.x;
  field_out[idx +   size_dev + 3*size_dev*blockIdx.y + 12*size_dev].w = aux.y;


                             // 11.x component 
  aux.x=(float)a_11.x;
  t4=a_11.x-aux.x;
  aux.y=(float)t4;
  field_out[idx + 2*size_dev + 3*size_dev*blockIdx.y              ].x = aux.x;
  field_out[idx + 2*size_dev + 3*size_dev*blockIdx.y + 12*size_dev].x = aux.y;
                             // 11.y component 
  aux.x=(float)a_11.y;
  t4=a_11.y-aux.x;
  aux.y=(float)t4;
  field_out[idx + 2*size_dev + 3*size_dev*blockIdx.y              ].y = aux.x;
  field_out[idx + 2*size_dev + 3*size_dev*blockIdx.y + 12*size_dev].y = aux.y;
                             // 12.x component 
  aux.x=(float)a_12.x;
  t4=a_12.x-aux.x;
  aux.y=(float)t4;
  field_out[idx + 2*size_dev + 3*size_dev*blockIdx.y              ].z = aux.x;
  field_out[idx + 2*size_dev + 3*size_dev*blockIdx.y + 12*size_dev].z = aux.y;
                             // 12.y component 
  aux.x=(float)a_12.y;
  t4=a_12.y-aux.x;
  aux.y=(float)t4;
  field_out[idx + 2*size_dev + 3*size_dev*blockIdx.y              ].w = aux.x;
  field_out[idx + 2*size_dev + 3*size_dev*blockIdx.y + 12*size_dev].w = aux.y;
  }



/*
================================================================== EXTERNAL C FUNCTION
*/



extern "C" void cuda_exp_su3(float step) 
  {
  #ifdef DEBUG_MODE
  printf("DEBUG: inside cuda_exp_su3 ...\n");
  #endif

  //Define constants and streams
  size_t gauge_field_size = sizeof(float4)*no_links*3;  
  size_t general_matrix_size   = sizeof(float2)*no_links*9;

  //Allocate device memory
  float2 *exp_field;
  cudaSafe(AT,cudaMalloc((void**)&exp_field, general_matrix_size), "cudaMalloc"); 

  cudaSafe(AT,cudaMemcpyToSymbol(f_aux_dev, &step, sizeof(float),  0,   cudaMemcpyHostToDevice), "cudaMemcpyToSymbol");

  //Setting block and grid sizes for the kernel
  dim3 ExpBlockDimension(NUM_THREADS);
  dim3 GridDimension(size/ExpBlockDimension.x, 4); //Run separately the four dimensions

  // exp_field=e^{i step*momenta} 
  ExponentiateKernel<<<GridDimension,ExpBlockDimension>>>(momenta_device, exp_field); 
  cudaCheckError(AT,"ExponentiateKernel");

  size_t offset_g;
  cudaSafe(AT,cudaBindTexture(&offset_g, gauge_texRef, gauge_field_device, gauge_field_size), "cudaBindTexture");
  offset_g/=sizeof(float4);

  // exp_field=exp_field*gauge_field_device
  MultiplyByFieldKernel<<<GridDimension,ExpBlockDimension>>>(exp_field, device_phases, offset_g);
  cudaCheckError(AT,"MultiplyVyFieldKernel");

  cudaSafe(AT,cudaUnbindTexture(gauge_texRef), "cudaUnbindTexture");

  // Unitarize gauge_filed_device
  ReunitarizeKernelD<<<GridDimension,ExpBlockDimension>>>(exp_field, gauge_field_device);
  cudaCheckError(AT,"ReunitarizeKernelD");

  //Free memory 
  cudaSafe(AT,cudaFree(exp_field), "cudaFree");

  #ifdef DEBUG_MODE
  printf("\tterminated cuda_exp_su3\n");
  #endif
  }

