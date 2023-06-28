CFLAGS?=-O3
override CFLAGS+=-MMD -MP -march=native
LDFLAGS?=

APP=test
SRC=main.c sudoku_solver.c

.PHONY:all
all:$(APP)

$(APP):$(patsubst %.c,%.o,$(SRC))
	$(CC) $(LDFLAGS) -o $@ $^

%.o:%.c
	$(CC) $(CFLAGS) -c $<

-include $(SRC:.c=.d)

clean:
	rm -f *.o *.d $(APP)
