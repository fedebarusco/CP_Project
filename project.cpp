/*
 * Prova inizio progetto calcolo parallelo
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cmath>
#include <cstdlib>
#include <mpi.h>
//#include <libhpm.h>
using namespace std;

typedef complex<double> number;

int main(int argc, char **argv) {

    if(argc != 4 || string(argv[1]) == "--help") {
        cout << "Usage: algorithm_type, input_file, output_file" << endl;
        cout << "algorithm: the fft algorithm to use. Valid options are radix-2, definition" << endl;
        cout << "input: path to file containing the input" << endl;
        cout << "output: path to the file in which the output will be saved to" << endl;
        exit(0);
    }

    string algorithm = argv[1];
    string in_file = argv[2];
    string out_file = argv[3];

    number *data, *ret;
    int length;		// Length of the vector to be transformed
    int rank, P;	// The id of the current process and the total number of processes

    // Initialize MPI Environment
    if(MPI_Init(&argc, &argv) != MPI_SUCCESS){
        cerr << "An error occured while initializing MPI!" << endl;
        exit(0);
    }

    // Get the number of processes
    if(MPI_Comm_size(MPI_COMM_WORLD, &P) != MPI_SUCCESS){
        cerr << "Error while getting the number of processes" << endl;
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_UNKNOWN);
    }

    // Get process id
    if(MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS){
        cerr << "Error while getting proccess id" << endl;
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_UNKNOWN);
    }

    // P0 loads the data and tells the other how much data there is
    if(rank == 0) loadData(in_file, data, length);
    MPI_Bcast(&length, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /*
     * #TODO: ora bisogna implementare i due algoritmi che voglia creare
     *      radix-2 che alla fine è colley e takey
     *      definition, che si basa sulla definizione     *
     */

    // Run the chosen algorithm
    if(algorithm == "radix-2"){
        fft_radix_2(data, length, rank, P);
        if(rank == 0)	// P0 is the only one with the complete (and transformed) vector
            saveData(out_file, data, length);
    }
    else if(algorithm == "definition"){
        ret = fft_definition(data, length, rank, P);
        if(rank == 0){  // P0 is the only one with the complete (and transformed) vector
            saveData(out_file, ret, length);
            delete[] ret;
        }
    }
    else{
        cerr << "Algorithm not supported" << endl;
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_ARG);
    }

    delete[] data;

    // Close the MPI Environment and exit
    MPI_Finalize();

    return 0;
}