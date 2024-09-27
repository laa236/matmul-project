#include<stdio.h>
#include<stdalign.h>
//#include<immintrin.h> // For SSE intrinsics
//#include<string.h>
const char* dgemm_desc = "My awesome dgemm.";
#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 128)
// a cache line is 64 bytes, or 8 doubles
#define CACHE_ALIGN __declspec(align(16))
#endif

/* matricies are labeled ROW x COL
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/

// J = columns of C
// 
void inline basic_dgemm(const int lda, const int M, const int N, const int K,
                 const double* restrict A, const double* restrict B, double* restrict C)
{
    int i, j, k;
    double storeBCol[K];
    for (j = 0; j < N; ++j) {   //EVERY ITERATION IS COLUMN OF B AND C

        for (k = 0; k < K; ++k) {   //GOES DOWN THE COLUMN OF B
            storeBCol[k] = B[j*lda+k];  //COPIES DOWN A COLUMN OF B
        }

        for (k = 0; k < K; ++k) {   //goes down the column of B, and row of A
            for (i = 0; i < M; ++i) {
                C[j*lda + i]+= A[k*lda+i] * storeBCol[k];
                //C = i -> goes down,                   j -> goes across
                //A = i -> goes down, k -> goes across,  
                //B =                 k -> goes down,   j -> goes across  
            }
        }
    }
}

void inline do_block(const int lda,
              const double* restrict A, const double* restrict B, double* restrict C,
              const int i, const int j, const int k)
{
    const int M = (i+BLOCK_SIZE > lda? lda-i : BLOCK_SIZE);
    const int N = (j+BLOCK_SIZE > lda? lda-j : BLOCK_SIZE);
    const int K = (k+BLOCK_SIZE > lda? lda-k : BLOCK_SIZE);
    basic_dgemm(lda, M, N, K,
                A + i + k*lda, B + k + j*lda, C + i + j*lda);
}

void square_dgemm(const int M, const double* restrict A, const double* restrict B, double* restrict C)
{
    const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0);
    int bi, bj, bk;
    for (bi = 0; bi < n_blocks; ++bi) {
        const int i = bi * BLOCK_SIZE;
        for (bj = 0; bj < n_blocks; ++bj) {
            const int j = bj * BLOCK_SIZE;
            for (bk = 0; bk < n_blocks; ++bk) {
                const int k = bk * BLOCK_SIZE;
                do_block(M, A, B, C, i, j, k);
            }
        }
    }
}
