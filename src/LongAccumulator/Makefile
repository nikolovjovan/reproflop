# C++ compiler
CC		= g++
CFLAGS	= -std=c++14 -O2
SRC		= LongAccumulator.cpp test.cpp
TARGET	= test

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm -rf $(TARGET)