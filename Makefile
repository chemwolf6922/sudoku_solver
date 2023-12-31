CFLAGS?=-O3
override CFLAGS+=-MMD -MP
LDFLAGS?=

APP=test
SRC=main.c sudoku_solver.c cpu_cycle_counter.c

.PHONY:all
all:$(APP)

$(APP):$(patsubst %.c,%.o,$(SRC))
	$(CC) $(LDFLAGS) -o $@ $^

%.o:%.c
	$(CC) $(CFLAGS) -c $<

-include $(SRC:.c=.d)

clean:
	rm -f *.o *.d $(APP)
