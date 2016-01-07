#ifndef __GOLD_H__
#define __GOLD_H__

#ifdef __cplusplus
extern "C" {
#endif

int* convolutionGold(int *, int, int, float *, int);
int* histogramGold(int *, int, int);
int* histogramGoldFloat(float *, int, int);

float* convolutionGoldFloat(float *, int, int, float *, int);

#ifdef __cplusplus
}
#endif

#endif
