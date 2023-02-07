#include <stdio.h>    
#include <stdlib.h>
#include <math.h>    
#include <iostream>
#include "HIP_Particle.h"
#include <hip/hip_runtime.h>



// Custom Class Particle which contains all info about the location and speed of a particle
// all data types 'double' to maximize precision
class Particle{
    public:
        double velocity[2];

        double current_position[2];

        double best_position[2];

        double best_position_value;

        double current_position_value;

       // Class Constructor
        Particle(){
            int n_dimensions = 2;
            for(int i = 0; i < n_dimensions; i++){
                velocity[i] = (double) rand() / RAND_MAX * 10; // ensures value is bounded by 10
                current_position[i] = (double) rand() / RAND_MAX * 10; // ^
                best_position[i] = current_position[i]; // best position is initial position during Particle construction/initialization

            }

            best_position_value = f(best_position[0], best_position[1]);
            current_position_value = f(current_position[0], current_position[1]);

            // EXPERIMENT B ONLY
            // best_position_value = g(best_position[0], best_position[1],best_position[2], best_position[3],
                                    // best_position[4], best_position[5]);

            
            // current_position_value = g(best_position[0], best_position[1],best_position[2], best_position[3],
                                    // best_position[4], best_position[5]);
        }

};

__host__ __device__ double f(double x, double y){
    return pow(x, 4) + pow(y, 4);
}



__global__ void update(Particle all_particles[], int n_particles, double global_position[], double global_position_value[], double c1, double c2, double w, double r1, double r2)
{

        int particle_number = blockIdx.x;
        int dimension_number = threadIdx.x;

        
        all_particles[particle_number].velocity[dimension_number] = w*all_particles[particle_number].velocity[dimension_number] 
                                                                + c1*r1*(all_particles[particle_number].best_position[dimension_number] 
                                                                - all_particles[particle_number].current_position[dimension_number]) 
                                                                + c2*r2*(global_position[dimension_number] 
                                                                - all_particles[particle_number].current_position[dimension_number]);
        

        all_particles[particle_number].current_position[dimension_number] = all_particles[particle_number].current_position[dimension_number] + all_particles[particle_number].velocity[dimension_number];



        __syncthreads();

        double new_value = f(all_particles[particle_number].current_position[0], all_particles[particle_number].current_position[1]);

        if (new_value <= all_particles[particle_number].best_position_value){
            all_particles[particle_number].best_position_value = new_value;
            all_particles[particle_number].best_position[dimension_number] = all_particles[particle_number].current_position[dimension_number];
        }
        


        if (all_particles[particle_number].best_position_value <= *global_position_value){
            *global_position_value = all_particles[particle_number].best_position_value;
            global_position[dimension_number] = all_particles[particle_number].best_position[dimension_number];
        }
        
        
}


int main(){

    int n_particles = 20;
    Particle all_particles[n_particles];

    Particle *d_particles;
    hipMalloc(&d_particles, n_particles * sizeof(Particle));
    hipMemcpy(d_particles, all_particles, n_particles * sizeof(Particle), hipMemcpyHostToDevice);

    double global_position_value = DBL_MAX;
    double global_position[2] = {0,0};

  
    
    for(int i = 0; i < n_particles; i++){
    
        if (all_particles[i].best_position_value < global_position_value){
            global_position_value = all_particles[i].best_position_value;
            global_position[0] = all_particles[i].best_position[0];
            global_position[1] = all_particles[i].best_position[1];
        }
    }


    double *d_global_position_value;
    double *d_global_position;

    hipMalloc(&d_global_position_value, 1*sizeof(double));
	hipMalloc(&d_global_position, 2*sizeof(double));

    double c1 = 1.49618;
    double c2 = 1.49618;
    double w = 0.7298;

    double r1 = (double) rand() / RAND_MAX;
    double r2 = (double) rand() / RAND_MAX;


    for(int j = 0; j < 100; j++){
        std::cout<< "Iteration " << j+1 << "\n";
        hipMemcpy(d_global_position_value, &global_position_value,  1*sizeof(double), hipMemcpyHostToDevice);
	    hipMemcpy(d_global_position, global_position, 2*sizeof(double), hipMemcpyHostToDevice); 
        
        update<<<n_particles, 2>>>(d_particles, n_particles, d_global_position, d_global_position_value, c1, c2, w, r1, r2);

        hipMemcpy(&global_position_value, d_global_position_value,  1 * sizeof(double), hipMemcpyDeviceToHost);
        hipMemcpy(global_position, d_global_position,  2 * sizeof(double), hipMemcpyDeviceToHost);
        hipMemcpy(all_particles, d_particles,  n_particles * sizeof(Particle), hipMemcpyDeviceToHost);

        std::cout << "Minimum Value :" << global_position_value << "\n";
    }
 
    std::cout << "PSO Found Best Solution at: " << (double) global_position[0] << ","<< global_position[1] << " which evaluates to " <<  global_position_value << "\n";
    return 0;
}