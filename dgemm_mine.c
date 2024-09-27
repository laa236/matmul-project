#include<stdio.h>
#include<stdalign.h>
//#include<immintrin.h> // For SSE intrinsics
#include<string.h>
#include<stdlib.h>
const char* dgemm_desc = "My awesome dgemm.";
#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 64)
// a cache line is 64 bytes, or 8 doubles
#define CACHE_ALIGN __declspec(align(16))
#endif

/* matricies are labeled ROW x COL
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/

/* BELOW IS MY DEBUGGER DO NOT DELETE

            char output[50];
            snprintf(output, 50, "%f",  storeBCol[k]);
            printf("%s ", output);

*/

// J = columns of C
// 
void inline basic_dgemm(const int lda, const int M, const int N, const int K,
                 double* restrict A, double* restrict B, double* restrict C)
{
    int i, j, k;
    //double storeBCol[BLOCK_SIZE];

    for (j = 0; j < BLOCK_SIZE; ++j) {   //EVERY ITERATION IS COLUMN OF B AND C
        for (k = 0; k < BLOCK_SIZE; ++k) {   //goes down the column of B, and row of A
            for (i = 0; i < BLOCK_SIZE; ++i) {
                C[j*lda + i] += A[k*BLOCK_SIZE+i] * B[j*BLOCK_SIZE+k];
                //C = i -> goes down,                   j -> goes across
                //A = i -> goes down, k -> goes across,  
                //B =                 k -> goes down,   j -> goes across  
            }
        }
    }
}

void inline do_block_mine(const int lda,
              const double* restrict A, const double* restrict B, double* restrict C,
              const int i, const int j, const int k, double* restrict temp_matrix_A, double* restrict temp_matrix_B)
{
    const int M = (i+BLOCK_SIZE > lda? lda-i : BLOCK_SIZE);
    const int N = (j+BLOCK_SIZE > lda? lda-j : BLOCK_SIZE);
    const int K = (k+BLOCK_SIZE > lda? lda-k : BLOCK_SIZE);
    //creates the sizes of the block
    // REMINDER: matrices are labeled M x N, where M is the number of rows (vertical measurement)
    // A will be generated as M * K
    // B will be generated as K * N
    // C will be generated as K * K

    memset(temp_matrix_A, 0, BLOCK_SIZE * BLOCK_SIZE * sizeof(double));
    memset(temp_matrix_B, 0, BLOCK_SIZE * BLOCK_SIZE * sizeof(double));
    // zeroes temp matrices

    // all the variables here refrence the dimensions of the original block
    // always going down column for cache hits
    for (int ak = 0; ak < K; ++ak) {
        for (int am = 0; am < M; ++am) {
            temp_matrix_A[ak * BLOCK_SIZE + am] = A[(k + ak) * lda + (i + am)];
        }
    }
    for (int bn = 0; bn < N; ++bn) {
        for (int bk = 0; bk < K; ++bk) {
            temp_matrix_B[bn * BLOCK_SIZE + bk] = B[(j + bn) * lda + (k + bk)];
        }
    }                                   

    basic_dgemm(lda, M, N, K,
                temp_matrix_A, temp_matrix_B, C + i + j * lda);
}

void square_dgemm(const int M, const double* restrict A, const double* restrict B, double* restrict C)
{
    const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0);
    int bi, bj, bk;
    alignas(16) double* temp_matrix_A;
    alignas(16) double* temp_matrix_B;
    posix_memalign((void**) &temp_matrix_A, 16, BLOCK_SIZE * BLOCK_SIZE * sizeof(double));
    posix_memalign((void**) &temp_matrix_B, 16, BLOCK_SIZE * BLOCK_SIZE * sizeof(double));

    for (bi = 0; bi < n_blocks; ++bi) {
        const int i = bi * BLOCK_SIZE;
        for (bj = 0; bj < n_blocks; ++bj) {
            const int j = bj * BLOCK_SIZE;
            for (bk = 0; bk < n_blocks; ++bk) {
                const int k = bk * BLOCK_SIZE;
                do_block_mine(M, A, B, C, i, j, k, temp_matrix_A, temp_matrix_B);

                // REMINDER: matrix stuff is labeled Aij
                // where i = row, j = column
                // so increasing j = going to the right
            }
        }
    }

    free(temp_matrix_A);
    free(temp_matrix_B);
}
