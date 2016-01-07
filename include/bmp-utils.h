#ifndef __BMPFUNCS_H__
#define __BMPFUNCS_H__

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned char uchar;

int*   readBmp(const char *, int *, int*);

float* readBmpFloat(const char *, int*, int*);

void writeBmp(int *, const char *, int, int, const char*),
     writeBmpFloat(float *, const char *, int, int, const char*);

#ifdef __cplusplus
}
#endif

#endif
