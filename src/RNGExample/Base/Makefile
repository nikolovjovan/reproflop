CC		= g++
CFLAGS	= -std=c++14 -pthread -O2
SRC		= RNGExample.cpp
TARGET	= RNGExample

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm -rf $(TARGET)