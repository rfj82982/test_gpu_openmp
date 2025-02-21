#include <iostream>
#include <omp.h>

int main() {
    const int X = 4, Y = 3, Z = 2;
    int data[X][Y][Z];

    // Get the number of OpenMP target devices
    int num_devices = omp_get_num_devices();
    std::cout << "Number of OpenMP devices available: " << num_devices << "\n";

    // Initialize data on the host
    for (int i = 0; i < X; ++i) {
        for (int j = 0; j < Y; ++j) {
            for (int k = 0; k < Z; ++k) {
                data[i][j][k] = i + j + k;
            }
        }
    }

    // Check if there are devices available
    if (num_devices > 0) {
        // Offload the computation to the GPU using a collapse(3) clause
        //#pragma omp target teams distribute parallel for collapse(3) map(tofrom: data[0:X][0:Y][0:Z])
        #pragma omp target teams distribute parallel for collapse(3) 
        for (int i = 0; i < X; ++i) {
            for (int j = 0; j < Y; ++j) {
                for (int k = 0; k < Z; ++k) {
                    data[i][j][k] *= 2; // Double each element
                }
            }
        }
    } else {
        std::cout << "No GPU devices available, computation will not be offloaded.\n";
        for (int i = 0; i < X; ++i) {
            for (int j = 0; j < Y; ++j) {
                for (int k = 0; k < Z; ++k) {
                    data[i][j][k] *= 2; // Fallback computation on the CPU
                }
            }
        }
    }

    // Print the results on the host
    std::cout << "Data after processing:\n";
    for (int i = 0; i < X; ++i) {
        for (int j = 0; j < Y; ++j) {
            for (int k = 0; k < Z; ++k) {
                std::cout << "data[" << i << "][" << j << "][" << k << "] = " << data[i][j][k] << "\n";
            }
        }
    }

    std::cout << "Hello from CPU and GPU with collapse(3)!\n";
    return 0;
}

