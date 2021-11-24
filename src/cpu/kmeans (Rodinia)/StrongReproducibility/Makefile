# C++ compiler
LACCDIR		= ../../LongAccumulator
CC			= g++
CC_FLAGS	= -std=c++14 -fopenmp -lm -O2 -I$(LACCDIR)
SRC			= cluster.cpp kmeans.cpp kmeans_clustering.cpp $(LACCDIR)/LongAccumulator.cpp
TARGET		= kmeans

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC) Makefile
	$(CC) $(CC_FLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
	rm -f output_*.log