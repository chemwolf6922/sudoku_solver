#ifndef SUDOKU_SOLVER_H
#define SUDOKU_SOLVER_H

#include <stdint.h>

typedef struct
{
    int numbers[9*9];   /** 0 means empty */
} sudoku_t;

void sudoku_solver_prepare();
int64_t sudoku_solver_solve(sudoku_t* s);

#endif

