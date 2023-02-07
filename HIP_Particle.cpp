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

// The function we are trying to optimize
// This is a host and device function since we're using it in both the kernel and the host functions
__host__ __device__ double f(double x, double y){
    // EXPERIMENT A 
        return pow(x, 2) + pow(y, 2);

    // EXPERIMENT C 
        // return -1*cos(x)*cos(y)*exp(-(pow((x-M_PI),2) + pow((y-M_PI),2)));
    
    // EXPERIMENT D
        //return pow(x, 0.5) + pow(y, 0.5); // pass decimals into pow function, not fractions
    
    // EXPERIMENT E 
        //return !((x<0) ^ (y<0));

    // EXPERIMENT F
        // if((x < 0) ^ (y<0)){
            // return 0;
        // } 
        // return abs(x)+abs(y);
}

// EXPERIMENT B
__host__ __device__ double g(double a, double b, double c, double d, double e, double f){
    return pow(a, 2) + pow(b, 2) + pow(c, 2) + pow(d, 2) + pow(e, 2) + pow(f, 2);
}




// Kernel to update the velocity and position of each dimension of each particle
__global__ void update(Particle all_particles[], double global_position[], double c1, double c2, double w, double r1, double r2, int n_dimensions)
{
        
        int particle_number = threadIdx.x / n_dimensions;
        int dimension_number = threadIdx.x % n_dimensions;

        
        all_particles[particle_number].velocity[dimension_number] = w*all_particles[particle_number].velocity[dimension_number] 
                                                                + c1*r1*(all_particles[particle_number].best_position[dimension_number] 
                                                                - all_particles[particle_number].current_position[dimension_number]) 
                                                                + c2*r2*(global_position[dimension_number] 
                                                                - all_particles[particle_number].current_position[dimension_number]);
        

        all_particles[particle_number].current_position[dimension_number] = all_particles[particle_number].current_position[dimension_number] + all_particles[particle_number].velocity[dimension_number];


}


// Kernel to calculate the value of the particle given its new position
__global__ void calculate(Particle all_particles[])
{
        int particle_number = blockIdx.x;

        double new_value = f(all_particles[particle_number].current_position[0], all_particles[particle_number].current_position[1]);
        
        // EXPERIMENT B ONLY
        // double new_value = g(all_particles[particle_number].current_position[0], all_particles[particle_number].current_position[1],
                            // all_particles[particle_number].current_position[2], all_particles[particle_number].current_position[3],
                            // all_particles[particle_number].current_position[4], all_particles[particle_number].current_position[5]);



        all_particles[particle_number].current_position_value = new_value;

}

// Kernel to update the global minimum if the value is a new-found minima
__global__ void compare(Particle all_particles[], double global_position[], double global_position_value[], int n_dimensions)
{
        //__shared__ double buffer[2];
        //buffer[0] = global_position[0];
        //buffer[1] = global_position[1];
        __syncthreads();

        //int particle_number = threadIdx.x / 2;
        //int dimension_number = threadIdx.x % 2;
        int particle_number = blockIdx.x;

      
        if (all_particles[particle_number].current_position_value <= all_particles[particle_number].best_position_value){
            all_particles[particle_number].best_position_value = all_particles[particle_number].current_position_value;
            for(int j = 0; j < n_dimensions; j++){
                all_particles[particle_number].best_position[j] = all_particles[particle_number].current_position[j];
            }
        }
        

        if (all_particles[particle_number].best_position_value <= *global_position_value){
            *global_position_value = all_particles[particle_number].best_position_value;
            for(int j = 0; j < n_dimensions; j++){
                global_position[j] = all_particles[particle_number].best_position[j];
            }
        }
        
        
        //global_position[0] = buffer[0];
        //global_position[1] = buffer[1];

}

// Shared Memory Version of previous 'compare' kernel.
// This kernel does not get invoked.
__global__ void parallel_compare(Particle all_particles[], double global_position[], double global_position_value[])
{
        __shared__ double buffer[2];
        buffer[0] = global_position[0];
        buffer[1] = global_position[1];
        

        int particle_number = blockIdx.x;
        int dimension_number = threadIdx.x;
        //int particle_number = threadIdx.x / 2;
        //int dimension_number = threadIdx.x % 2;
        //int particle_number = blockIdx.x;

        if (all_particles[particle_number].current_position_value <= all_particles[particle_number].best_position_value){
            all_particles[particle_number].best_position_value = all_particles[particle_number].current_position_value;
            all_particles[particle_number].best_position[dimension_number] = all_particles[particle_number].current_position[dimension_number];
        }

        __syncthreads();

        if (all_particles[particle_number].best_position_value <= *global_position_value){
            *global_position_value = all_particles[particle_number].best_position_value;
            buffer[dimension_number] = all_particles[particle_number].best_position[dimension_number];
        }
        
        __syncthreads();
        global_position[0] = buffer[0];
        global_position[1] = buffer[1];

}



int main(){
    // initialize timing variables
    hipEvent_t start, stop; 
    float time; 
    hipEventCreate(&start); 
    hipEventCreate(&stop); 

    // for reproducibility
    srand(1);

    // Size of the Swarm - Hyperparamter  
    // We use a size of 100 Particles to emphasize the GPU speedup 
    int n_particles = 100;
    int n_dimensions = 2;

    Particle all_particles[n_particles];


	//allocate our memory on GPU 
    Particle *d_particles;
    hipMalloc(&d_particles, n_particles * sizeof(Particle));
    hipMemcpy(d_particles, all_particles, n_particles * sizeof(Particle), hipMemcpyHostToDevice);



    double global_position_value = DBL_MAX;
    double global_position[n_dimensions];

    // Make sure the current global minimum is the minimum amongst particles
    // This is important because the initial global minimum is included in the first velocity update
    for(int i = 0; i < n_particles; i++){
            if (all_particles[i].best_position_value < global_position_value){
                global_position_value = all_particles[i].best_position_value;
                for(int j = 0; j < n_dimensions; j++){
                    global_position[j] = all_particles[i].best_position[j];
                }
        }
    }
  

    
    //allocate our memory on GPU 

    double *d_global_position_value;
    double *d_global_position;
    

    hipMalloc(&d_global_position_value, 1*sizeof(double));
	hipMalloc(&d_global_position, n_dimensions*sizeof(double));


    // Initialization of hyperparameters
    double c1 = 1.49618;
    double c2 = 1.49618;
    double w = 0.7298;

    double r1 = (double) rand() / RAND_MAX;
    double r2 = (double) rand() / RAND_MAX;


    int j = 0;

    // Start recording
    hipEventRecord(start, 0);  
    
    // EXPERIMENT A, B, D, E, F
        while(global_position_value > 0 && j < 100000){

    // EXPERIMENT C 
        // while(global_position_value != -1){       
        
        j++;

        // Memory Transfer to Device (GPU)
        hipMemcpy(d_global_position_value, &global_position_value,  1*sizeof(double), hipMemcpyHostToDevice);
	    hipMemcpy(d_global_position, global_position, n_dimensions*sizeof(double), hipMemcpyHostToDevice); 
        
        update<<<1, n_dimensions*n_particles>>>(d_particles, d_global_position, c1, c2, w, r1, r2, n_dimensions);
        calculate<<<n_particles, 1>>>(d_particles);
        compare<<<n_particles, 1>>>(d_particles, d_global_position, d_global_position_value, n_dimensions);
        //parallel_compare<<<n_particles, 2>>>(d_particles, d_global_position, d_global_position_value);

        // Transfer Memory back from Device (GPU) to Host (Local PC)
        hipMemcpy(&global_position_value, d_global_position_value,  1 * sizeof(double), hipMemcpyDeviceToHost);
        hipMemcpy(global_position, d_global_position,  n_dimensions * sizeof(double), hipMemcpyDeviceToHost);
        hipMemcpy(all_particles, d_particles,  n_particles * sizeof(Particle), hipMemcpyDeviceToHost);


    }

    // Stop recording
    hipEventRecord(stop, 0); 
    hipEventSynchronize(stop); 
    hipEventElapsedTime(&time, start, stop); 


    std::cout << "Time: "<< time/1000 << " seconds \n";
    std::cout << "In " << j << " iterations, PSO Found Best Solution at: " << (double) global_position[0] << ","<< global_position[1] << " which evaluates to " <<  global_position_value << "\n";
    
    // EXPERIMENT B ONLY
    // std::cout << "In " << j << " iterations, PSO Found Best Solution at: " << global_position[0] << ","<< global_position[1] << "," << global_position[2] << ","<< global_position[3] << ","
    // << global_position[4] << ","<< global_position[5] << " which evaluates to " <<  global_position_value << "\n";
    
    return 0;
}