# C++ compiler
CC			= g++
CC_FLAGS	= -fopenmp -lm -O2
SRC			= cluster.c kmeans.c kmeans_clustering.c
TARGET		= kmeans

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC) Makefile
	$(CC) $(CC_FLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
	rm -f output_*.log