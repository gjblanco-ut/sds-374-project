CXX = g++
CFLAGS = -g -Wall

TARGET = main

all: $(TARGET)

#add file.o rule
myfile.o: myfile.cpp myfile.h
	g++ -c myfile.cpp $(CFLAGS) -o myfile.o

linear_algebra.o: linear_algebra.h linear_algebra.cpp 
	g++ -c linear_algebra.cpp $(CFLAGS) -o linear_algebra.o

neural_net.o: neural_net.h neural_net.cpp 
	g++ -c neural_net.cpp $(CFLAGS) -o neural_net.o

# add the file to each of the following lists
FILES_O = linear_algebra.o neural_net.o

main.o: main.cpp
	g++ -c $(CFLAGS) main.cpp

$(TARGET): main.o $(FILES_O)
	g++ main.o $(CFLAGS) -o $(TARGET)

clean:
	rm -f *.o $(TARGET)