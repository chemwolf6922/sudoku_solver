#define _GNU_SOURCE

#include <stdio.h>
#include <inttypes.h>
#include <time.h>
#include <unistd.h>
#include "sudoku_solver.h"
#include "cpu_cycle_counter.h"

static uint64_t now_ns()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ((uint64_t)ts.tv_sec)*1000000000 + (uint64_t)ts.tv_nsec;
}

int main(int argc, char const *argv[])
{
    sudoku_t s = {
        .numbers = {8,0,0, 0,0,0, 0,0,0,
                    0,0,3, 6,0,0, 0,0,0,
                    0,7,0, 0,9,0, 2,0,0,
                    
                    0,5,0, 0,0,7, 0,0,0,
                    0,0,0, 0,4,5, 7,0,0,
                    0,0,0, 1,0,0, 0,3,0,
                    
                    0,0,1, 0,0,0, 0,6,8,
                    0,0,8, 5,0,0, 0,1,0,
                    0,9,0, 0,0,0, 4,0,0}
    };
    sudoku_solver_prepare();
    int fd = cpu_cycle_counter_open();
    cpu_cycle_counter_reset(fd);
    uint64_t start_ns = now_ns();
    int64_t n_iter = sudoku_solver_solve(&s);
    uint64_t time_past = now_ns() - start_ns;
    long long cpu_cycles = cpu_cycle_counter_get_result(fd);
    close(fd);
    if(n_iter < 0)
    {
        printf("Can not solve.\n");
        return 1;
    }
    printf("Solved in %"PRId64" iterations [%"PRIu64" ns, %lld cycles]\n", n_iter, time_past, cpu_cycles);
    printf("%"PRIu64"ns per iteration, %lldcycles per iteration\n",time_past/n_iter, cpu_cycles/n_iter);
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
