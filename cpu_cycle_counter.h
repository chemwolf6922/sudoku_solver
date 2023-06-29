#ifndef CPU_CYCLE_COUNTER_H
#define CPU_CYCLE_COUNTER_H

int cpu_cycle_counter_open();

int cpu_cycle_counter_reset(int fd);
long long cpu_cycle_counter_get_result(int fd);

#endif
