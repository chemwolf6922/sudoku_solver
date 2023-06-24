#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include "sudoku_solver.h"

typedef struct sudoku_entry_s
{
    struct sudoku_entry_s* prev;
    struct sudoku_entry_s* next;
    int8_t number;         /** 0-8, -1 means empty */
    int8_t row;
    int8_t column;
    int8_t block;
} sudoku_entry_t;

typedef struct
{
    sudoku_entry_t entries[9*9];
    sudoku_entry_t* empty_entry;
    int rows[9];
    int columns[9];
    int blocks[9];
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
    sudoku_entry_t* empty_entry = NULL;
    for(int i=0; i<9*9; i++)
    {
        if(input->numbers[i] < 0 || input->numbers[i] > 9)  /** invalid number */
            return -1;
        s->entries[i].number = input->numbers[i] - 1;
        s->entries[i].row = WHICH_ROW(i);
        s->entries[i].column = WHICH_COLUMN(i);
        s->entries[i].block = WHICH_BLOCK(s->entries[i].row, s->entries[i].column);
        if(s->entries[i].number == -1)
        {
            if(empty_entry == NULL)
                s->empty_entry = &s->entries[i];
            else
                empty_entry->next = &s->entries[i];
            s->entries[i].prev = empty_entry;
            empty_entry = &s->entries[i];
        }
    }
    /** construct bit maps */
    for(int i = 0; i < 9; i++)
    {
        for(int j=0; j < 9; j++)
        {
            /** rows */
            int n = s->entries[LOCATE_ROW_ITEM(i,j)].number;
            if(n >= 0)
            {   
                if(s->rows[i] & 1<<n)   /** duplicate numbers in a row */
                    return -1;
                s->rows[i] |= 1<<n;  
            }
            /** columns */
            n = s->entries[LOCATE_COLUMN_ITEM(i,j)].number;
            if(n >= 0)
            {   
                if(s->columns[i] & 1<<n)   /** duplicate numbers in a column */
                    return -1;
                s->columns[i] |= 1<<n;  
            }
            /** blocks */
            n = s->entries[LOCATE_BLOCK_ITEM(i,j)].number;
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

static int solve_next(sudoku_internal_t* s)
{
    int min_i,min_j;
    possibility_t* p = NULL;
    s->n_iter ++;
    /** find the empty slot with the least possibilities */
    sudoku_entry_t* entry = s->empty_entry;
    sudoku_entry_t* min_entry = NULL;
    while(entry)
    {
        /** get the combined bit map */
        int bits = s->rows[entry->row] | s->columns[entry->column] | s->blocks[entry->block];
        /** find arg_min */
        if(!p || p->number_of_possibilities > cache.possibilities[bits].number_of_possibilities)
        {
            min_entry = entry;
            min_i = entry->row;
            min_j = entry->column;
            p = &cache.possibilities[bits];
        }
        entry = entry->next;
    }
    /** check if the sudoku is finished */
    if(!p)
        return 0;
    /** try all possibilities */
    for(int k=0;k<p->number_of_possibilities;k++)
    {
        /** update s */
        int n = p->possibilities[k];
        min_entry->number = n;
        /** unlink entry */
        if(min_entry->prev)
            min_entry->prev->next = min_entry->next;
        else
            s->empty_entry = min_entry->next;
        if(min_entry->next)
            min_entry->next->prev = min_entry->prev;  
        s->rows[min_i] |= 1<<n;
        s->columns[min_j] |= 1<<n;
        s->blocks[WHICH_BLOCK(min_i,min_j)] |= 1<<n;
        /** solve next */
        if(solve_next(s) == 0)
            return 0;           /** propagate back success */
        /** restore s if this does not work */
        min_entry->number = -1;
        /** relink entry */
        if(min_entry->prev)
            min_entry->prev->next = min_entry;
        else
            s->empty_entry = min_entry;
        if(min_entry->next)
            min_entry->next->prev = min_entry;  
        s->rows[min_i] ^= 1<<n;
        s->columns[min_j] ^= 1<<n;
        s->blocks[WHICH_BLOCK(min_i,min_j)] ^= 1<<n;
    }
    /** no luck */
    return -1;
}

static void convert_result_back(sudoku_internal_t* s, sudoku_t* output)
{
    for(int i=0; i<9*9; i++)
        output->numbers[i] = s->entries[i].number + 1;
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
