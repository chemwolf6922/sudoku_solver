#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <immintrin.h>
#include "sudoku_solver.h"

typedef struct
{
    int n_numbers;
    int16_t numbers[96];    /** only 9*9 are used */
    int16_t rows[32];       /** only 9 are used */
    int16_t columns[32];    /** only 9 are used */
    int16_t blocks[32];     /** only 9 are used */
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

#define REVERSE_PARAMS32(v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,v21,v22,v23,v24,v25,v26,v27,v28,v29,v30,v31)\
    (v31),(v30),(v29),(v28),(v27),(v26),(v25),(v24),\
    (v23),(v22),(v21),(v20),(v19),(v18),(v17),(v16),\
    (v15),(v14),(v13),(v12),(v11),(v10), (v9), (v8),\
     (v7), (v6), (v5), (v4), (v3), (v2), (v1), (v0)

static inline int arg_min_possibility_avx512(sudoku_internal_t* s)
{
    /** load luts */
    __m512i row_lut = _mm512_loadu_si512(s->rows);
    __m512i col_lut = _mm512_loadu_si512(s->columns);
    __m512i blk_lut = _mm512_loadu_si512(s->blocks);

    /** 0-31 */
    /** lookup row for the first 32 numbers */
    __m512i row_index = _mm512_set_epi16(REVERSE_PARAMS32(0,0,0,0,0,0,0,0,0,
                                                          1,1,1,1,1,1,1,1,1,
                                                          2,2,2,2,2,2,2,2,2,
                                                          3,3,3,3,3));
    __m512i row = _mm512_permutex2var_epi16(row_lut, row_index, row_lut);
    /** lookup col for the first 32 numbers */
    __m512i col_index = _mm512_set_epi16(REVERSE_PARAMS32(0,1,2,3,4,5,6,7,8,
                                                          0,1,2,3,4,5,6,7,8,
                                                          0,1,2,3,4,5,6,7,8,
                                                          0,1,2,3,4));
    __m512i col = _mm512_permutex2var_epi16(col_lut, col_index, col_lut);
    /** lookup blk for the first 32 numbers */
    __m512i blk_index = _mm512_setr_epi16(REVERSE_PARAMS32(0,0,0,1,1,1,2,2,2,
                                                           0,0,0,1,1,1,2,2,2,
                                                           0,0,0,1,1,1,2,2,2,
                                                           3,3,3,4,4));
    __m512i blk = _mm512_permutex2var_epi16(blk_lut, blk_index, blk_lut);

    /** or total results for the first 32 numbers */
    __m512i total = _mm512_or_si512(row, col);
    total = _mm512_or_si512(total, blk);
    /** count ones */
    __m512i ones_0_31 = _mm512_popcnt_epi16(total);

    /** 32 - 63 */
    row_index = _mm512_set_epi16(REVERSE_PARAMS32(3,3,3,3,
                                                  4,4,4,4,4,4,4,4,4,
                                                  5,5,5,5,5,5,5,5,5,
                                                  6,6,6,6,6,6,6,6,6,
                                                  7));
    row = _mm512_permutex2var_epi16(row_lut, row_index, row_lut);
    col_index = _mm512_set_epi16(REVERSE_PARAMS32(5,6,7,8,
                                                  0,1,2,3,4,5,6,7,8,
                                                  0,1,2,3,4,5,6,7,8,
                                                  0,1,2,3,4,5,6,7,8,
                                                  0));
    col = _mm512_permutex2var_epi16(col_lut, col_index, col_lut);
    blk_index = _mm512_set_epi16(REVERSE_PARAMS32(4,5,5,5,
                                                  3,3,3,4,4,4,5,5,5,
                                                  3,3,3,4,4,4,5,5,5,
                                                  6,6,6,7,7,7,8,8,8,
                                                  6));
    blk = _mm512_permutex2var_epi16(blk_lut, blk_index, blk_lut);

    total = _mm512_or_si512(row, col);
    total = _mm512_or_si512(total, blk);
    __m512i ones_32_63 = _mm512_popcnt_epi16(total);

    /** 64 - 80(95) */
    row_index = _mm512_set_epi16(REVERSE_PARAMS32(7,7,7,7,7,7,7,7,
                                                  8,8,8,8,8,8,8,8,8,
                                                  0,0,0,0,0,0,0,0,0,
                                                  0,0,0,0,0,0));
    row = _mm512_permutex2var_epi16(row_lut, row_index, row_lut);
    col_index = _mm512_set_epi16(REVERSE_PARAMS32(1,2,3,4,5,6,7,8,
                                                  0,1,2,3,4,5,6,7,8,
                                                  0,0,0,0,0,0,0,0,0,
                                                  0,0,0,0,0,0));
    col = _mm512_permutex2var_epi16(col_lut, col_index, col_lut);
    blk_index = _mm512_set_epi16(REVERSE_PARAMS32(6,6,7,7,7,8,8,8,
                                                  6,6,6,7,7,7,8,8,8,
                                                  0,0,0,0,0,0,0,0,0,
                                                  0,0,0,0,0,0));
    blk = _mm512_permutex2var_epi16(blk_lut, blk_index, blk_lut);

    total = _mm512_or_si512(row, col);
    total = _mm512_or_si512(total, blk);
    __m512i ones_64_80 = _mm512_popcnt_epi16(total);

    /** make all taken or invalid slot to -1 */
    __m512i numbers = _mm512_loadu_si512(&s->numbers[0]);
    __mmask32 empty_mask = _mm512_cmpeq_epi16_mask(numbers, _mm512_set1_epi16(-1));
    ones_0_31 = _mm512_mask_blend_epi16(empty_mask, _mm512_set1_epi16(-1), ones_0_31);
    numbers = _mm512_loadu_si512(&s->numbers[32]);
    empty_mask = _mm512_cmpeq_epi16_mask(numbers, _mm512_set1_epi16(-1));
    ones_32_63 = _mm512_mask_blend_epi16(empty_mask, _mm512_set1_epi16(-1), ones_32_63);
    numbers = _mm512_loadu_si512(&s->numbers[64]);
    empty_mask = _mm512_cmpeq_epi16_mask(numbers, _mm512_set1_epi16(-1));
    ones_64_80 = _mm512_mask_blend_epi16(empty_mask, _mm512_set1_epi16(-1), ones_64_80);

    /** find the arg max */
    __m512i max = _mm512_max_epi16(ones_0_31, ones_32_63);
    max = _mm512_max_epi16(max, ones_64_80);
    /** fold 32 -> 16 */
    max = _mm512_max_epi16(max, _mm512_permutex2var_epi64(max, _mm512_set_epi64(4,5,6,7,0,1,2,3), max));
    /** fold 16 -> 8 */
    max = _mm512_max_epi16(max, _mm512_permutex2var_epi64(max, _mm512_set_epi64(2,3,0,1,6,7,4,5), max));
    /** fold 8 -> 4 */
    max = _mm512_max_epi16(max, _mm512_alignr_epi8(max, max, 8));
    /** fold 4 -> 2 */
    max = _mm512_max_epi16(max, _mm512_alignr_epi8(max, max, 4));
    /** fold 2 -> 1 */
    max = _mm512_max_epi16(max, _mm512_alignr_epi8(max, max, 2));
    /** max at this point are all the same values */
    __mmask32 mask_0_31 = _mm512_cmpeq_epi16_mask(max, ones_0_31);
    if(mask_0_31)
        return __builtin_ctz(mask_0_31);
    __mmask32 mask_32_63 = _mm512_cmpeq_epi16_mask(max, ones_32_63);
    if(mask_32_63)
        return 32 + __builtin_ctz(mask_32_63);
    __mmask32 mask_64_80 = _mm512_cmpeq_epi16_mask(max, ones_64_80);
    return 64 + __builtin_ctz(mask_64_80);
}

static int solve_next(sudoku_internal_t* s)
{
    s->n_iter ++;
    /** check if the sudoku is finished */
    if(s->n_numbers == 9*9)
        return 0;
    int pos = arg_min_possibility_avx512(s);
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
