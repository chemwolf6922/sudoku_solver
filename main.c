#include <stdio.h>
#include <inttypes.h>
#include "sudoku_solver.h"

int main(int argc, char const *argv[])
{
    sudoku_t s = {
        .numbers = {2,0,1, 8,0,0, 0,0,4,
                    8,9,0, 3,0,0, 2,6,1,
                    0,6,7, 1,0,9, 0,0,5,
                    
                    0,0,8, 0,0,6, 0,0,0,
                    0,0,3, 5,0,0, 6,0,0,
                    0,0,2, 7,4,3, 0,9,8,
                    
                    0,0,0, 0,0,0, 0,1,9,
                    5,0,9, 0,3,2, 0,0,6,
                    0,0,0, 0,1,7, 4,5,2}
    };
    int64_t n_iter = sudoku_solver_solve(&s);
    if(n_iter < 0)
    {
        printf("Can not solve.\n");
        return 1;
    }
    printf("Solved in %"PRId64" iterations\n", n_iter);
    for(int i=0;i<9;i++)
    {
        for(int j=0;j<9;j++)
        {
            putchar(s.numbers[i*9+j]+'0');
            putchar(' ');
        }
        putchar('\n');
    }
    return 0;
}
