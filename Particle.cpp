#include <stdio.h>    
#include <stdlib.h>
#include <math.h>    
#include <iostream>
#include "Particle.h"
#include <ctime>


// Custom Class Particle which contains all info about the location and speed of a particle
// all data types 'double' to maximize precision
class Particle{
    public:
        double velocity[2];

        double current_position[2];

        double best_position[2];

        double best_position_value;

        // Class Constructor
        Particle(){
            int n_dimensions = 2;
            for(int i = 0; i < n_dimensions; i++){
                velocity[i] = (double) rand() / RAND_MAX * 10; // ensures value is bounded by 10
                current_position[i] = (double) rand() / RAND_MAX * 10; // ^
                best_position[i] = current_position[i]; // best position is initial position during Particle construction/initialization

            }

            best_position_value = f(best_position[0], best_position[1]);

            // EXPERIMENT B ONLY
            // best_position_value = g(best_position[0], best_position[1],best_position[2], best_position[3], best_position[4], best_position[5]);
        }

};


// The function we are trying to optimize
double f(double x, double y){
    // EXPERIMENT A 
        return pow(x, 2) + pow(y, 2);

    // EXPERIMENT C 
        //return -1*cos(x)*cos(y)*exp(-(pow((x-M_PI),2) + pow((y-M_PI),2)));
    
    // EXPERIMENT D
        //return pow(x, 0.5) + pow(y, 0.5); // pass decimals into pow function, not fractions
    
    // EXPERIMENT E 
        // return !((x<0) ^ (y<0));

    // EXPERIMENT F
        // if((x < 0) ^ (y<0)){
            // return 0;
        // } 
        // return abs(x)+abs(y);
}

// EXPERIMENT B
double g(double a, double b, double c, double d, double e, double f){
    return pow(a, 2) + pow(b, 2) + pow(c, 2) + pow(d, 2) + pow(e, 2) + pow(f, 2);
}



// Static global initialization of hyperparameters
double c1 = 1.49618;
double c2 = 1.49618;
double w = 0.7298;

double r1 = (double) rand() / RAND_MAX; // bounded by 1
double r2 = (double) rand() / RAND_MAX; // ^



void update(Particle all_particles[], int n_particles, int n_dimensions){
    for(int i = 0; i < n_particles; i++){

        // Update Velocity of each dimension
        for(int j = 0; j < n_dimensions; j++){
            all_particles[i].velocity[j] = w*all_particles[i].velocity[j] + c1*r1*(all_particles[i].best_position[j] - all_particles[i].current_position[j]) 
                                            + c2*r2*(global_position[j] - all_particles[i].current_position[j]);
            

            // Update position for each dimension
            all_particles[i].current_position[j] = all_particles[i].current_position[j] + all_particles[i].velocity[j];
        }

        // Calculate the value we are now at after updating the current position
        double new_value = f(all_particles[i].current_position[0], all_particles[i].current_position[1]);
        

        // EXPERIMENT B ONLY
        // double new_value = g(all_particles[i].current_position[0], all_particles[i].current_position[1],
                            // all_particles[i].current_position[2], all_particles[i].current_position[3],
                            // all_particles[i].current_position[4], all_particles[i].current_position[5]);



        // update each Particle's best location
        if (new_value < all_particles[i].best_position_value){
            all_particles[i].best_position_value = new_value;
            for(int j = 0; j < n_dimensions; j++){
                all_particles[i].best_position[j] = all_particles[i].current_position[j];

            }
        }

        // update global minimum if Particle has found a better position
        if (all_particles[i].best_position_value < global_position_value){
            global_position_value = all_particles[i].best_position_value;
            for(int j = 0; j < n_dimensions; j++){
                global_position[j] = all_particles[i].best_position[j];
            }
        }
    }
}



int main(){

    // for reproducibility
    srand(1);

    // Size of the Swarm - Hyperparamter  
    // We use a size of 100 Particles to emphasize the GPU speedup 
    int n_particles = 100;
    int n_dimensions = 2;

    Particle all_particles[n_particles];

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


    int j = 0;
    
    // Start Timing
    clock_t start = clock();
    
    // EXPERIMENT A, B, D, E, F
        while(global_position_value > 0 && j < 100000){
    // EXPERIMENT C 
        //while(global_position_value != -1){
    
        j++;
        update(all_particles, n_particles, n_dimensions);
    }
    

    // Stop Timing
    clock_t end = clock(); 
    double time = double(end - start) / CLOCKS_PER_SEC; // [1] 	

    std::cout << "Time: "<< time << " seconds \n";
    std::cout << "In " << j << " iterations, PSO Found Best Solution at: " << (double) global_position[0] << ","<< global_position[1] << " which evaluates to " <<  global_position_value << "\n";


    // EXPERIMENT B ONLY
    // std::cout << "In " << j << " iterations, PSO Found Best Solution at: " << global_position[0] << ","<< global_position[1] << "," << global_position[2] << ","<< global_position[3] << ","
    // << global_position[4] << ","<< global_position[5] << " which evaluates to " <<  global_position_value << "\n";

    return 0;
}
