SOURCES=$(wildcard *.c)
HEADERS=$(wildcard *.h)
OBJECTS=$(wildcard *.o)
CFLAGS=-w -gdwarf-4 -g3 -g

build:
	gcc $(CFLAGS) -c $(SOURCES)
	gcc -o NeuronSampler $(CFLAGS) $(OBJECTS)