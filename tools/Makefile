# Builds h3probe against a sibling checkout of antirez/h3.c.
H3 ?= $(HOME)/projects/github/antirez/h3.c

h3probe: h3probe.c $(H3)/libh3.a
	clang -std=c11 -O2 -Wall -Wextra -I$(H3) -o $@ $< $(H3)/libh3.a \
	  -framework Foundation -framework Metal \
	  -framework MetalPerformanceShaders \
	  -framework MetalPerformanceShadersGraph \
	  -framework Accelerate -licucore -lm

clean:
	rm -f h3probe
