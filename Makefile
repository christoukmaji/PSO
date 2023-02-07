C = hipcc
CFLAGS = -std=c++11

all: HIP_Particle

HIP_Particle: HIP_Particle.o 
	$(C) $(CFLAGS) -o HIP_Particle.exe HIP_Particle.o 
HIP_Particle.o: HIP_Particle.cpp
	$(C) $(CFLAGS) -c HIP_Particle.cpp -o HIP_Particle.o
clean:
	rm -f HIP_Particle.exe *.dat *.o
