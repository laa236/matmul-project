/*
    Written for Project 1, CS 5220
    Fall 2024

    Lawrence Atienza / Jay Chawrey / Shivam Thakkar
    laa236 / jpc239 / skt55
*/

#include<stdalign.h>
    // for aligning
#include<immintrin.h> 
    // For SIMD
#include<string.h>
    // For memcpy
//#include<stdlib.h>
//#include<stdio.h>
    //above two were for debugging, not needed anymore
const char* dgemm_desc = "My awesome dgemm.";
#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 32)
    //determined through trial and error
#endif

/* matricies are labeled ROW x COL
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/

/* BELOW IS MY DEBUGGER DO NOT DELETE MY BABY

            char output[50];
            snprintf(output, 50, "%f",  storeBCol[k]);
            printf("%s ", output);

*/

//this is the kernel, only works on input matrices that are square and have a dimension divisible by 16
//A and B are temp matrices with size BLOCK_SIZE * BLOCK_SIZE
void inline basic_dgemm(const int lda, double* restrict A, double* restrict B, double* restrict C)
{
    for (int j = 0; j < BLOCK_SIZE; ++j) {   // Iterate over columns of B and C
        for (int k = 0; k < BLOCK_SIZE; ++k) { // Iterate down the column of B and row of A
            // Load the value from B into every index of vector
            __m512d b_val = _mm512_set1_pd(B[j * BLOCK_SIZE + k]);

            // Process 16 items at a time
            // each 512 bit vector can hold 8 doubles
            for (int i = 0; i < BLOCK_SIZE; i += 16) {
                //every line is here is doubled to hint to the compiler to 2 SIMD ops per cycle
                __m512d a_vals = _mm512_load_pd(&A[k * BLOCK_SIZE + i]);
                __m512d a_vals2 = _mm512_load_pd(&A[k * BLOCK_SIZE + i + 8]);
                    //load 16 values from column of A
                __m512d c_vals = _mm512_load_pd(&C[j * lda + i]);
                __m512d c_vals2 = _mm512_load_pd(&C[j * lda + i + 8]);
                    //load 16 values from column of C
                c_vals = _mm512_fmadd_pd(a_vals, b_val, c_vals);
                c_vals2 = _mm512_fmadd_pd(a_vals2, b_val, c_vals2);
                    //do the fused mult add, put back into C
                _mm512_storeu_pd(&C[j * lda + i], c_vals);
                _mm512_storeu_pd(&C[j * lda + i + 8], c_vals2);
                    //Store the results back to C
            }
        }
    }
}

//creates the temp matrices using the "anchor point" from square_dgemm
//then passes them into the kernel to do math
void inline do_block_mine(const int lda,
              const double* restrict A, const double* restrict B, double* restrict C,
              const int i, const int j, const int k, double* restrict temp_matrix_A, double* restrict temp_matrix_B)
{
    const int M = (i+BLOCK_SIZE > lda? lda-i : BLOCK_SIZE);
    const int N = (j+BLOCK_SIZE > lda? lda-j : BLOCK_SIZE);
    const int K = (k+BLOCK_SIZE > lda? lda-k : BLOCK_SIZE);
    // creates the sizes of the block
    // while the actual matrices are always BLOCK_SIZE * BLOCK_SIZE, we dont want to copy extra stuff
    // that could cause math problems
    /*
        REMINDER: matrices are labeled M x N, where M is the number of rows (vertical measurement)
        A will be generated as M * K
        B will be generated as K * N
        C will be generated as K * K
    */

    memset(temp_matrix_A, 0, BLOCK_SIZE * BLOCK_SIZE * sizeof(double));
    memset(temp_matrix_B, 0, BLOCK_SIZE * BLOCK_SIZE * sizeof(double));
    // zeroes temp matrices

    // all the variables here refrence the dimensions of the original block
    // using memcpy to copy columns
    // only copies what it needs to, nothing more. extra empty space in the temp matrices are padding
    for (int ak = 0; ak < K; ++ak) {
        memcpy(&temp_matrix_A[ak * BLOCK_SIZE], &A[(k + ak) * lda + i], M * sizeof(double));
        // for (int am = 0; am < M; ++am) {
        //     temp_matrix_A[ak * BLOCK_SIZE + am] = A[(k + ak) * lda + (i + am)];
        // }
    }
    for (int bn = 0; bn < N; ++bn) {
        memcpy(&temp_matrix_B[bn * BLOCK_SIZE], &B[(j + bn) * lda + k], K * sizeof(double));
        // for (int bk = 0; bk < K; ++bk) {
        //     temp_matrix_B[bn * BLOCK_SIZE + bk] = B[(j + bn) * lda + (k + bk)];
        // }
    }                                   

    //passes to kernel
    //M N K are not needed since we use temp matrices
    basic_dgemm(lda, temp_matrix_A, temp_matrix_B, C + i + j * lda);
}

//handles determining the index of the upper left member of each block, the "anchor point"
//also generates the temp matrices for copy optimization
//passes both to do_block_mine
void square_dgemm(const int M, const double* restrict A, const double* restrict B, double* restrict C)
{
    const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0);
        //figures out number of blocks in each direction, prof's code
    int bi, bj, bk;
    alignas(16) double* temp_matrix_A;
    alignas(16) double* temp_matrix_B;
        //creates pointers for temp matrices
    posix_memalign((void**) &temp_matrix_A, 64, BLOCK_SIZE * BLOCK_SIZE * sizeof(double));
    posix_memalign((void**) &temp_matrix_B, 64, BLOCK_SIZE * BLOCK_SIZE * sizeof(double)); 
        //creates aligned temp matrices

    for (bi = 0; bi < n_blocks; ++bi) {
        const int i = bi * BLOCK_SIZE;
        for (bj = 0; bj < n_blocks; ++bj) {
            const int j = bj * BLOCK_SIZE;
            for (bk = 0; bk < n_blocks; ++bk) {
                const int k = bk * BLOCK_SIZE;
                do_block_mine(M, A, B, C, i, j, k, temp_matrix_A, temp_matrix_B);
                    //passes index info + temp pointers

                // REMINDER: matrix stuff is labeled Aij
                // where i = row, j = column
                // so increasing j = going to the right
            }
        }
    }

    free(temp_matrix_A);
    free(temp_matrix_B);
        //I forgot to free once and the machine blew up in my face lol so here we go
}
