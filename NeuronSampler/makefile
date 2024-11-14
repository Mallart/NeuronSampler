SOURCES=$(wildcard *.c)
HEADERS=$(wildcard *.h)
OBJECTS=$(wildcard *.o)
CFLAGS=-w

build:
	gcc $(CFLAGS) -c $(SOURCES)
	gcc -o NeuronSampler $(CFLAGS) $(OBJECTS)