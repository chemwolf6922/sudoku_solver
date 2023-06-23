#ifndef SUDOKU_SOLVER_H
#define SUDOKU_SOLVER_H

#include <stdint.h>

typedef struct
{
    int numbers[9*9];   /** 0 means empty */
} sudoku_t;

int64_t sudoku_solver_solve(sudoku_t* s);

#endif

