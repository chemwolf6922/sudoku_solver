#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include "sudoku_solver.h"

typedef struct
{
    int n_numbers;
    int8_t numbers[128];    /** only 9*9 are used */
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
    bool built;
    possibility_t possibilities[1<<9];
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

static inline int arg_min_possibility(sudoku_internal_t* s)
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

static int solve_next(sudoku_internal_t* s)
{
    s->n_iter ++;
    /** check if the sudoku is finished */
    if(s->n_numbers == 9*9)
        return 0;
    int pos = arg_min_possibility(s);
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
