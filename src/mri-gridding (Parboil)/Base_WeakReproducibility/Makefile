# C++ compiler
CC			= g++
CC_FLAGS	= -std=c++14 -fopenmp -lm -O2
SRC			= main.cpp CPU_kernels.cpp
TARGET		= mri-gridding

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC) Makefile
	$(CC) $(CC_FLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
	rm -f output_*.log