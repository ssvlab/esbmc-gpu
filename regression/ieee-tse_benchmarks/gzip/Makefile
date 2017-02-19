allfile: allfile.o
	gcc -o allfile allfile.o

allfile.o: allfile.c crypt.h getopt.h gzip.h lzw.h revision.h tailor.h config.h
	gcc -DHAVE_CONFIG_H -I. -c allfile.c

clean:
	rm -f allfile.o

all: clean allfile
