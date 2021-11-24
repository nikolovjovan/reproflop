# C++ compiler
CC			= g++
CC_FLAGS	= -std=c++14 -fopenmp -lm -O2
SRC			= cluster.cpp kmeans.cpp kmeans_clustering.cpp
TARGET		= kmeans

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC) Makefile
	$(CC) $(CC_FLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
	rm -f output_*.log