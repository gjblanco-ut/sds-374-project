CXX = g++
# CFLAGS = -g -Wall

# TARGET = main

# all: $(TARGET)

# #add file.o rule
# myfile.o: myfile.cpp myfile.h
# 	g++ -c myfile.cpp $(CFLAGS) -o myfile.o

# linear_algebra.o: linear_algebra.h linear_algebra.cpp 
# 	g++ -c linear_algebra.cpp $(CFLAGS) -o linear_algebra.o

# neural_net.o: neural_net.h neural_net.cpp 
# 	g++ -c neural_net.cpp $(CFLAGS) -o neural_net.o

# # add the file to each of the following lists
# FILES_O = linear_algebra.o neural_net.o myfile.o

# main.o: main.cpp
# 	g++ -c $(CFLAGS) main.cpp

# $(TARGET): main.o $(FILES_O)
# 	g++ main.o $(CFLAGS) -o $(TARGET)

# clean:
# 	rm -f *.o $(TARGET)
# set non-optional compiler flags here

CXXFLAGS += -std=c++11 -Wall -Wextra -pedantic-errors

# set non-optional preprocessor flags here
# eg. project specific include directories
CPPFLAGS += 

# find cpp files in subdirectories
SOURCES := $(shell find . -name '*.cpp')

# find headers
HEADERS := $(shell find . -name '*.h')

OUTPUT := main

# Everything depends on the output
all: $(OUTPUT)

# The output depends on sources and headers
$(OUTPUT): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $(OUTPUT) $(SOURCES)

run:
	make main && time ./main -e 1000000000 -l 5 -s 100 -r 0.05 -d ../../mnist

clean:
	$(RM) $(OUTPUT)