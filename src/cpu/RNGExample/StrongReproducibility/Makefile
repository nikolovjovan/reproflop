LACCDIR = ../../LongAccumulator
CC		= g++
CFLAGS	= -std=c++14 -frounding-math -pthread -O2 -I$(LACCDIR)
SRC		= RNGExample.cpp $(LACCDIR)/LongAccumulator.cpp
TARGET	= RNGExample

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm -rf $(TARGET)