#ifndef SUDOKU_SOLVER_H
#define SUDOKU_SOLVER_H

typedef struct
{
    int numbers[9*9];   /** 0 means empty */
} sudoku_t;

int sudoku_solver_solve(sudoku_t* s);

#endif

