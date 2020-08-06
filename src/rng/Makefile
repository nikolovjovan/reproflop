CC		= g++
CFLAGS	= -std=c++14 -pthread
SRC		= rng.cpp
TARGET	= rng

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm -rf $(TARGET)