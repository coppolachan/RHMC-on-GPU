#ifndef DSFMT_PARAMS11213_H
#define DSFMT_PARAMS11213_H

#define DSFMT_POS1	42
#define DSFMT_SL1	37
#define DSFMT_SL2	3
#define DSFMT_SR1	7
#define DSFMT_SR2	16
#define DSFMT_MSK1	UINT64_C(0xffdbfdbfdfbb7ffe)
#define DSFMT_MSK2	UINT64_C(0xfbf7ff7ffbef3df7)
#define DSFMT_MSK32_1	0xffdbfdbfU
#define DSFMT_MSK32_2	0xdfbb7ffeU
#define DSFMT_MSK32_3	0xfbf7ff7fU
#define DSFMT_MSK32_4	0xfbef3df7U
#define DSFMT_PCV1	UINT64_C(0x0000000000000001)
#define DSFMT_PCV2	UINT64_C(0x00032a9a00000000)
#define DSFMT_IDSTR \
	"dSFMT-11213:42-37-3-7-16:ffdbfdbfdfbb7ffe-fbf7ff7ffbef3df7"


/* PARAMETERS FOR ALTIVEC */
#if defined(__APPLE__)	/* For OSX */
    #define ALTI_SL1 	(vector unsigned int)(5, 5, 5, 5)
    #define ALTI_SL1_PERM \
	(vector unsigned char)(4,5,6,7,28,28,28,28,12,13,14,15,0,1,2,3)
    #define ALTI_SL1_MSK \
	(vector unsigned int)(0xffffffe0U,0x00000000U,0xffffffe0U,0x00000000U)
    #define ALTI_SL2_PERM \
	(vector unsigned char)(3,4,5,6,7,29,29,29,11,12,13,14,15,0,1,2)
    #define ALTI_SR1 \
	(vector unsigned int)(DSFMT_SR1, DSFMT_SR1, DSFMT_SR1, DSFMT_SR1)
    #define ALTI_SR1_MSK \
	(vector unsigned int)(0x01dbfdbfU,0xdfbb7ffeU,0x01f7ff7fU,0xfbef3df7U)
    #define ALTI_SR2_PERM \
	(vector unsigned char)(18,18,0,1,2,3,4,5,18,18,8,9,10,11,12,13)
    #define ALTI_PERM \
	(vector unsigned char)(8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7)
    #define ALTI_LOW_MSK \
	(vector unsigned int)(DSFMT_LOW_MASK32_1, DSFMT_LOW_MASK32_2, \
		DSFMT_LOW_MASK32_1, DSFMT_LOW_MASK32_2)
    #define ALTI_HIGH_CONST \
	(vector unsigned int)(DSFMT_HIGH_CONST32, 0, DSFMT_HIGH_CONST32, 0)
#else	/* For OTHER OSs(Linux?) */
    #define ALTI_SL1 	{5, 5, 5, 5}
    #define ALTI_SL1_PERM \
	{4,5,6,7,28,28,28,28,12,13,14,15,0,1,2,3}
    #define ALTI_SL1_MSK \
	{0xffffffe0U,0x00000000U,0xffffffe0U,0x00000000U}
    #define ALTI_SL2_PERM \
	{3,4,5,6,7,29,29,29,11,12,13,14,15,0,1,2}
    #define ALTI_SR1 \
	{DSFMT_SR1, DSFMT_SR1, DSFMT_SR1, DSFMT_SR1}
    #define ALTI_SR1_MSK \
	{0x01dbfdbfU,0xdfbb7ffeU,0x01f7ff7fU,0xfbef3df7U}
    #define ALTI_SR2_PERM \
	{18,18,0,1,2,3,4,5,18,18,8,9,10,11,12,13}
    #define ALTI_PERM \
	{8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7}
    #define ALTI_LOW_MSK \
	{DSFMT_LOW_MASK32_1, DSFMT_LOW_MASK32_2, \
		DSFMT_LOW_MASK32_1, DSFMT_LOW_MASK32_2}
    #define ALTI_HIGH_CONST \
	{DSFMT_HIGH_CONST32, 0, DSFMT_HIGH_CONST32, 0}
#endif

#endif /* DSFMT_PARAMS11213_H */
