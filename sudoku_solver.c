#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <immintrin.h>
#include "sudoku_solver.h"

#include <stdio.h>

typedef struct
{
    int n_numbers;
    int8_t numbers[96];    /** only 9*9 are used */
    int16_t rows[16];       /** only 9 are used */
    int16_t columns[16];    /** only 9 are used */
    int16_t blocks[16];     /** only 9 are used */
    int64_t n_iter;
} sudoku_internal_t;

typedef struct
{
    int8_t number_of_possibilities;
    int8_t possibilities[9];
} possibility_t;

typedef struct
{
    possibility_t possibilities[1<<9];
    bool built;
} sudoku_cache_t;

static sudoku_cache_t cache = {
    .built = false
};

#define LOCATE_ROW_ITEM(row,item) ((row)*9+(item))
#define LOCATE_COLUMN_ITEM(column,item) ((item)*9+(column))
#define LOCATE_BLOCK_ITEM(block,item) (((block)/3)*27 + ((block)%3)*3 + ((item)/3)*9 + (item)%3)
#define WHICH_ROW(n) ((n)/9)
#define WHICH_COLUMN(n) ((n)%9)
#define WHICH_BLOCK(row,column) ((row)/3*3+(column)/3)

static void build_cache()
{
    if(cache.built)
        return;
    memset(&cache, 0, sizeof(sudoku_cache_t));
    cache.built = true;
    for(int i=0; i<1<<9; i++)
    {
        /** iterate bits */
        for(int j=0;j<9;j++)
        {
            if(!(i & 1<<j))
            {
                cache.possibilities[i].possibilities[cache.possibilities[i].number_of_possibilities] = j;
                cache.possibilities[i].number_of_possibilities ++;
            }
        }
    }
}

static int check_and_convert_input(sudoku_t* input, sudoku_internal_t* s)
{
    memset(s, 0, sizeof(sudoku_internal_t));
    for(int i=0; i<9*9; i++)
    {
        if(input->numbers[i] < 0 || input->numbers[i] > 9)  /** invalid number */
            return -1;
        s->numbers[i] = input->numbers[i] - 1;
        if(s->numbers[i] != -1)
            s->n_numbers++;
    }
    /** construct bit maps */
    for(int i = 0; i < 9; i++)
    {
        for(int j=0; j < 9; j++)
        {
            /** rows */
            int n = s->numbers[LOCATE_ROW_ITEM(i,j)];
            if(n >= 0)
            {   
                if(s->rows[i] & 1<<n)   /** duplicate numbers in a row */
                    return -1;
                s->rows[i] |= 1<<n;  
            }
            /** columns */
            n = s->numbers[LOCATE_COLUMN_ITEM(i,j)];
            if(n >= 0)
            {   
                if(s->columns[i] & 1<<n)   /** duplicate numbers in a column */
                    return -1;
                s->columns[i] |= 1<<n;  
            }
            /** blocks */
            n = s->numbers[LOCATE_BLOCK_ITEM(i,j)];
            if(n >= 0)
            {   
                if(s->blocks[i] & 1<<n)   /** duplicate numbers in a block */
                    return -1;
                s->blocks[i] |= 1<<n;  
            }
        }   
    }
    return 0;
}

static inline int arg_min_possibility_baseline(sudoku_internal_t* s)
{
    /** find the empty slot with the least possibilities */
    int arg_min = -1;
    int min = INT32_MAX;
    for(int i=0;i<9;i++)
    {
        for(int j=0;j<9;j++)
        {
            if(s->numbers[LOCATE_ROW_ITEM(i,j)] == -1)
            {
                int bits = s->rows[i] | s->columns[j] | s->blocks[WHICH_BLOCK(i,j)];
                int n = cache.possibilities[bits].number_of_possibilities;
                if(n < min)
                {
                    min = n;
                    arg_min = LOCATE_ROW_ITEM(i,j);
                }
            }
        }
    }
    return arg_min;
}

static inline int arg_min_possibility_avx2(sudoku_internal_t* s)
{
    /** arrange row into two lookup tables */
    __m256i row_lut = _mm256_loadu_si256((__m256i_u*)s->rows);
    __m256i row_lut_lo = _mm256_and_si256(row_lut, _mm256_set1_epi16(0x00FF)); 
    row_lut_lo = _mm256_packus_epi16(row_lut_lo, row_lut_lo);
    row_lut_lo = _mm256_permute4x64_epi64(row_lut_lo, _MM_SHUFFLE(3,1,2,0));
    __m256i row_lut_hi = _mm256_srli_epi16(row_lut, 8);
    row_lut_hi = _mm256_packus_epi16(row_lut_hi, row_lut_hi);
    row_lut_hi = _mm256_permute4x64_epi64(row_lut_hi, _MM_SHUFFLE(3,1,2,0));
    /** arrange col into two lookup tables */
    __m256i col_lut = _mm256_loadu_si256((__m256i_u*)s->columns);
    __m256i col_lut_lo = _mm256_and_si256(col_lut, _mm256_set1_epi16(0x00FF)); 
    col_lut_lo = _mm256_packus_epi16(col_lut_lo, col_lut_lo);
    col_lut_lo = _mm256_permute4x64_epi64(col_lut_lo, _MM_SHUFFLE(3,1,2,0));
    __m256i col_lut_hi = _mm256_srli_epi16(col_lut, 8);
    col_lut_hi = _mm256_packus_epi16(col_lut_hi, col_lut_hi);
    col_lut_hi = _mm256_permute4x64_epi64(col_lut_hi, _MM_SHUFFLE(3,1,2,0));
    /** arrange blk into two lookup tables */
    __m256i blk_lut = _mm256_loadu_si256((__m256i_u*)s->blocks);
    __m256i blk_lut_lo = _mm256_and_si256(blk_lut, _mm256_set1_epi16(0x00FF)); 
    blk_lut_lo = _mm256_packus_epi16(blk_lut_lo, blk_lut_lo);
    blk_lut_lo = _mm256_permute4x64_epi64(blk_lut_lo, _MM_SHUFFLE(3,1,2,0));
    __m256i blk_lut_hi = _mm256_srli_epi16(blk_lut, 8);
    blk_lut_hi = _mm256_packus_epi16(blk_lut_hi, blk_lut_hi);
    blk_lut_hi = _mm256_permute4x64_epi64(blk_lut_hi, _MM_SHUFFLE(3,1,2,0));

    /** 0-31 */
    /** lookup row for the first 32 numbers */
    __m256i row_index = _mm256_setr_epi8(0,0,0,0,0,0,0,0,0,
                                         1,1,1,1,1,1,1,1,1,
                                         2,2,2,2,2,2,2,2,2,
                                         3,3,3,3,3);
    __m256i row_lo = _mm256_shuffle_epi8(row_lut_lo, row_index);
    __m256i row_hi = _mm256_shuffle_epi8(row_lut_hi, row_index);
    /** lookup col for the first 32 numbers */
    __m256i col_index = _mm256_setr_epi8(0,1,2,3,4,5,6,7,8,
                                         0,1,2,3,4,5,6,7,8,
                                         0,1,2,3,4,5,6,7,8,
                                         0,1,2,3,4);
    __m256i col_lo = _mm256_shuffle_epi8(col_lut_lo, col_index);
    __m256i col_hi = _mm256_shuffle_epi8(col_lut_hi, col_index);
    /** lookup blk for the first 32 numbers */
    __m256i blk_index = _mm256_setr_epi8(0,0,0,1,1,1,2,2,2,
                                         0,0,0,1,1,1,2,2,2,
                                         0,0,0,1,1,1,2,2,2,
                                         3,3,3,4,4);
    __m256i blk_lo = _mm256_shuffle_epi8(blk_lut_lo, blk_index);
    __m256i blk_hi = _mm256_shuffle_epi8(blk_lut_hi, blk_index);

    /** or total results for the first 32 numbers */
    __m256i total_lo = _mm256_or_si256(row_lo, col_lo);
    total_lo = _mm256_or_si256(total_lo, blk_lo);
    __m256i total_hi = _mm256_or_si256(row_hi, col_hi);
    total_hi = _mm256_or_si256(total_hi, blk_hi);

    /** count ones */
    __m256i ones_lut = _mm256_setr_epi8(0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
    __m256i total_lo_lo = _mm256_and_si256(total_lo, _mm256_set1_epi8(0x0F));
    total_lo_lo = _mm256_shuffle_epi8(ones_lut, total_lo_lo);
    __m256i total_lo_hi = _mm256_srli_epi16(total_lo, 4);
    total_lo_hi = _mm256_and_si256(total_lo_hi, _mm256_set1_epi8(0x0F));
    total_lo_hi = _mm256_shuffle_epi8(ones_lut, total_lo_hi);
    __m256i ones_0_31 = _mm256_add_epi8(total_lo_lo, total_lo_hi);
    ones_0_31 = _mm256_add_epi8(ones_0_31, total_hi);

    /** 32 - 63 */
    row_index = _mm256_setr_epi8(3,3,3,3,
                       4,4,4,4,4,4,4,4,4,
                       5,5,5,5,5,5,5,5,5,
                       6,6,6,6,6,6,6,6,6,
                       7);
    row_lo = _mm256_shuffle_epi8(row_lut_lo, row_index);
    row_hi = _mm256_shuffle_epi8(row_lut_hi, row_index);
    col_index = _mm256_setr_epi8(5,6,7,8,
                       0,1,2,3,4,5,6,7,8,
                       0,1,2,3,4,5,6,7,8,
                       0,1,2,3,4,5,6,7,8,
                       0);
    col_lo = _mm256_shuffle_epi8(col_lut_lo, col_index);
    col_hi = _mm256_shuffle_epi8(col_lut_hi, col_index);
    blk_index = _mm256_setr_epi8(4,5,5,5,
                       3,3,3,4,4,4,5,5,5,
                       3,3,3,4,4,4,5,5,5,
                       6,6,6,7,7,7,8,8,8,
                       6);
    blk_lo = _mm256_shuffle_epi8(blk_lut_lo, blk_index);
    blk_hi = _mm256_shuffle_epi8(blk_lut_hi, blk_index);

    total_lo = _mm256_or_si256(row_lo, col_lo);
    total_lo = _mm256_or_si256(total_lo, blk_lo);
    total_hi = _mm256_or_si256(row_hi, col_hi);
    total_hi = _mm256_or_si256(total_hi, blk_hi);

    total_lo_lo = _mm256_and_si256(total_lo, _mm256_set1_epi8(0x0F));
    total_lo_lo = _mm256_shuffle_epi8(ones_lut, total_lo_lo);
    total_lo_hi = _mm256_srli_epi16(total_lo, 4);
    total_lo_hi = _mm256_and_si256(total_lo_hi, _mm256_set1_epi8(0x0F));
    total_lo_hi = _mm256_shuffle_epi8(ones_lut, total_lo_hi);
    __m256i ones_32_63 = _mm256_add_epi8(total_lo_lo, total_lo_hi);
    ones_32_63 = _mm256_add_epi8(ones_32_63, total_hi);

    /** 64 - 80(95) */
    row_index = _mm256_setr_epi8(7,7,7,7,7,7,7,7,
                               8,8,8,8,8,8,8,8,8,
                               0,0,0,0,0,0,0,0,0,
                               0,0,0,0,0,0);
    row_lo = _mm256_shuffle_epi8(row_lut_lo, row_index);
    row_hi = _mm256_shuffle_epi8(row_lut_hi, row_index);
    col_index = _mm256_setr_epi8(1,2,3,4,5,6,7,8,
                               0,1,2,3,4,5,6,7,8,
                               0,0,0,0,0,0,0,0,0,
                               0,0,0,0,0,0);
    col_lo = _mm256_shuffle_epi8(col_lut_lo, col_index);
    col_hi = _mm256_shuffle_epi8(col_lut_hi, col_index);
    blk_index = _mm256_setr_epi8(6,6,7,7,7,8,8,8,
                               6,6,6,7,7,7,8,8,8,
                               0,0,0,0,0,0,0,0,0,
                               0,0,0,0,0,0);
    blk_lo = _mm256_shuffle_epi8(blk_lut_lo, blk_index);
    blk_hi = _mm256_shuffle_epi8(blk_lut_hi, blk_index);

    total_lo = _mm256_or_si256(row_lo, col_lo);
    total_lo = _mm256_or_si256(total_lo, blk_lo);
    total_hi = _mm256_or_si256(row_hi, col_hi);
    total_hi = _mm256_or_si256(total_hi, blk_hi);

    total_lo_lo = _mm256_and_si256(total_lo, _mm256_set1_epi8(0x0F));
    total_lo_lo = _mm256_shuffle_epi8(ones_lut, total_lo_lo);
    total_lo_hi = _mm256_srli_epi16(total_lo, 4);
    total_lo_hi = _mm256_and_si256(total_lo_hi, _mm256_set1_epi8(0x0F));
    total_lo_hi = _mm256_shuffle_epi8(ones_lut, total_lo_hi);
    __m256i ones_64_80 = _mm256_add_epi8(total_lo_lo, total_lo_hi);
    ones_64_80 = _mm256_add_epi8(ones_64_80, total_hi);

    /** make all taken or invalid slot to -1 */
    __m256i numbers = _mm256_loadu_si256((__m256i_u*)&s->numbers[0]);
    __m256i empty_mask = _mm256_cmpeq_epi8(numbers, _mm256_set1_epi8(-1));
    ones_0_31 = _mm256_blendv_epi8(_mm256_set1_epi8(-1), ones_0_31, empty_mask);
    numbers = _mm256_loadu_si256((__m256i_u*)&s->numbers[32]);
    empty_mask = _mm256_cmpeq_epi8(numbers, _mm256_set1_epi8(-1));
    ones_32_63 = _mm256_blendv_epi8(_mm256_set1_epi8(-1), ones_32_63, empty_mask);
    numbers = _mm256_loadu_si256((__m256i_u*)&s->numbers[64]);
    empty_mask = _mm256_cmpeq_epi8(numbers, _mm256_set1_epi8(-1));
    ones_64_80 = _mm256_blendv_epi8(_mm256_set1_epi8(-1), ones_64_80, empty_mask);

    /** find the arg max */
    __m256i max_mask = _mm256_cmpgt_epi8(ones_0_31, ones_32_63);
    __m256i max = _mm256_blendv_epi8(ones_32_63, ones_0_31, max_mask);
    __m256i max_index = _mm256_blendv_epi8(
        _mm256_setr_epi8(32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63),
        _mm256_setr_epi8( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31),
        max_mask
    );

    max_mask = _mm256_cmpgt_epi8(max, ones_64_80);
    max = _mm256_blendv_epi8(ones_64_80, max, max_mask);
    max_index = _mm256_blendv_epi8(
        _mm256_setr_epi8(64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        max_index,
        max_mask
    );
    /** fold 32 -> 16 */
    __m256i max_hi = _mm256_permute4x64_epi64(max, _MM_SHUFFLE(1,0,3,2));
    __m256i max_index_hi = _mm256_permute4x64_epi64(max_index, _MM_SHUFFLE(1,0,3,2));
    max_mask = _mm256_cmpgt_epi8(max, max_hi);
    max = _mm256_blendv_epi8(max_hi, max, max_mask);
    max_index = _mm256_blendv_epi8(max_index_hi, max_index, max_mask);
    /** fold 16 -> 8 */
    max_hi = _mm256_shuffle_epi32(max, _MM_SHUFFLE(1,0,3,2));
    max_index_hi = _mm256_shuffle_epi32(max_index, _MM_SHUFFLE(1,0,3,2));
    max_mask = _mm256_cmpgt_epi8(max, max_hi);
    max = _mm256_blendv_epi8(max_hi, max, max_mask);
    max_index = _mm256_blendv_epi8(max_index_hi, max_index, max_mask);
    /** fold 8 -> 4 */
    max_hi = _mm256_shuffle_epi32(max, _MM_SHUFFLE(2,3,0,1));
    max_index_hi = _mm256_shuffle_epi32(max_index, _MM_SHUFFLE(2,3,0,1));
    max_mask = _mm256_cmpgt_epi8(max, max_hi);
    max = _mm256_blendv_epi8(max_hi, max, max_mask);
    max_index = _mm256_blendv_epi8(max_index_hi, max_index, max_mask);
    /** fold 4 -> 2 */
    max_hi = _mm256_shufflelo_epi16(max, _MM_SHUFFLE(2,3,0,1));
    max_index_hi = _mm256_shufflelo_epi16(max_index, _MM_SHUFFLE(2,3,0,1));
    max_mask = _mm256_cmpgt_epi8(max, max_hi);
    max = _mm256_blendv_epi8(max_hi, max, max_mask);
    max_index = _mm256_blendv_epi8(max_index_hi, max_index, max_mask);
    /** 2 -> 1 */
    int8_t max_raw[32];
    uint8_t max_index_raw[32];
    _mm256_storeu_si256((__m256i_u*)max_raw, max);
    _mm256_storeu_si256((__m256i_u*)max_index_raw, max_index);
    if(max_raw[0] > max_raw[1])
        return max_index_raw[0];
    else
        return max_index_raw[1];
}

static int solve_next(sudoku_internal_t* s)
{
    s->n_iter ++;
    /** check if the sudoku is finished */
    if(s->n_numbers == 9*9)
        return 0;
    int pos = arg_min_possibility_avx2(s);
    int row = WHICH_ROW(pos);
    int column = WHICH_COLUMN(pos);
    int block = WHICH_BLOCK(row,column);
    int16_t bits = s->rows[row] | s->columns[column] | s->blocks[block];
    possibility_t* p = &cache.possibilities[bits];
    /** try all possibilities */
    for(int i=0;i<p->number_of_possibilities;i++)
    {
        /** update s */
        int n = p->possibilities[i];
        s->n_numbers ++;
        s->numbers[pos] = n;
        s->rows[row] |= 1<<n;
        s->columns[column] |= 1<<n;
        s->blocks[block] |= 1<<n;
        /** solve next */
        if(__builtin_expect(solve_next(s) == 0,0))
            return 0;           /** propagate back success */
        /** restore s if this does not work */
        s->n_numbers --;
        s->numbers[pos] = -1;
        s->rows[row] ^= 1<<n;
        s->columns[column] ^= 1<<n;
        s->blocks[block] ^= 1<<n;
    }
    /** no luck */
    return -1;
}

static void convert_result_back(sudoku_internal_t* s, sudoku_t* output)
{
    for(int i=0; i<9*9; i++)
        output->numbers[i] = s->numbers[i] + 1;
}

void sudoku_solver_prepare()
{
    build_cache();
}

int64_t sudoku_solver_solve(sudoku_t* input)
{
    build_cache();
    sudoku_internal_t s = {0};
    if(check_and_convert_input(input, &s) != 0)
        return -1;
    if(solve_next(&s) != 0)
        return -1;
    convert_result_back(&s, input);
    return s.n_iter;
}
