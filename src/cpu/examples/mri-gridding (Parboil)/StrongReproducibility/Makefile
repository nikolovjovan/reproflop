# C++ compiler
LACCDIR = ../../LongAccumulator
CC			= g++
CC_FLAGS	= -std=c++14 -fopenmp -lm -O2 -I$(LACCDIR)
SRC			= main.cpp CPU_kernels.cpp $(LACCDIR)/LongAccumulator.cpp
TARGET		= mri-gridding

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC) Makefile
	$(CC) $(CC_FLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
	rm -f output_*.log