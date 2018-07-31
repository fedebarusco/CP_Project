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

void fft_radix_2(number *&data, int n, int rank, int P);
int bit_reversal(int x, int num_bit);
void loadData(const string data_path, number* &buffer, int &length);
void saveData(const string data_path, number *buffer, int length);

int main(int argc, char **argv) {

    /*if(argc != 4 || string(argv[1]) == "--help") {
        cout << "Usage: algorithm_type, input_file, output_file" << endl;
        cout << "algorithm: the fft algorithm to use. Valid options are radix-2, definition" << endl;
        cout << "input: path to file containing the input" << endl;
        cout << "output: path to the file in which the output will be saved to" << endl;
        exit(0);
    }*/

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
     * #TODO: ora bisogna implementare i due algoritmi che vogliamo creare
     *      radix-2 che alla fine è colley e takey (per ora copiato spudoratamente dall'originale)
     *      definition, che si basa sulla definizione
     */

    fft_radix_2(data, length, rank, P);
    if(rank == 0)	// P0 is the only one with the complete (and transformed) vector
        saveData(out_file, data, length);

    delete[] data;

    // Close the MPI Environment and exit
    MPI_Finalize();

    return 0;
}



void loadData(const string data_path, number *&buffer, int &length) {

    fstream file(data_path.c_str(), ios_base::in);

    if(!file.is_open()){
        cerr << "Unable to open '" << data_path << "', quitting..." << endl;
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_ARG);	// Terminates all processes
    }

    // Read how many elements are stored in the file
    file >> length;
    buffer = new number[length];

    int count = 0;
    double r,i;
    file >> r;	// Read in one go the real and immaginary part
    file >> i;

    while(!file.eof()) {
        buffer[count] = number(r,i);
        count++;
        file >> r;	// Read the next couple, if the last values read were the last couple in the
        file >> i;	// file, than this read fails and the eof flag is set
    }

    file.close();
}

/*
 * Saves the vector in the specified file with the same convention used when reading from file in the function loadData
*/
void saveData(const string data_path, number *buffer, int length) {

    fstream file(data_path.c_str(), ios_base::out);

    if(!file.is_open()){
        cerr << "Unable to open '" << data_path << "', quitting..." << endl;
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_ARG);	// Terminates all processes
    }

    file << length << endl;
    for(int k = 0; k < length; k++) {
        file << buffer[k].real() << endl;
        file << buffer[k].imag() << endl;
    }

    file.flush();
    file.close();
}

void fft_radix_2(number *&data, int n, int rank, int P){

    if(rank == 0 && (P < 1 || P > n / 2)){	// We assume P <= n/2 and P power of 2
        cerr << "Please specify a number of processes between 1 and " << n/2 << ", quitting..." << endl;
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_ARG);	// Terminates all processes
    }

    //hpmInit(rank, "radix-2");
    const int length_description = 100;
    char description[length_description];

    //hpmStart(1, "Whole algorithm");

    int M = ceil((double)n / (double)P);
    int m = floor((double)n / (double)P);
    int threshold = n % P - 1;	// Index of the last process which has M data (all process with smaller indexes will have M data)
    int amount_data_M = M * (threshold + 1);	// Total amount of data stored in the first M processes
    int amount_M = threshold + 1;
    int num = (rank <= threshold) ? M : m;
    // Index of the first element of the group of the origina data associated to this process
    int base = (rank <= threshold) ? (M * rank) : (amount_data_M + (rank - amount_M) * m);
    if(rank > 0)	// Assuming P0 have the data and nobody else has anything
        data = new number[num];

    // Distribute the data to the processes
    //hpmStart(2, "Data distribution");
    if(rank == 0) {
        for(int k = 0; k <= threshold; k++)
            MPI_Send(data + k * M, M * sizeof(number), MPI_BYTE, k, k, MPI_COMM_WORLD);
        for(int k = threshold+1; k < P; k++)
            MPI_Send(data+amount_data_M+(k-threshold-1)*m, m * sizeof(number), MPI_BYTE, k, k, MPI_COMM_WORLD);
    }
    MPI_Recv(data, num * sizeof(number), MPI_BYTE, 0, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //hpmStop(2);

    // Start the computation
    number on = exp(number(0, 2 * M_PI / n));	// n-th radix of the unit
    int levels = log2(n);
    MPI_Request * req = new MPI_Request[num];
    int req_count = 0;
    number *temp_data = new number[2*num];
    int *indexes = new int[num];
    vector<bool> index_done(num, false);

    for(int d = 0; d < levels; d++) {
        int dist = n / pow((double)2, d+1);
        int d2 = pow((double)2, d);

        req_count = 0;
        for(int i = 0; i < index_done.size(); i++) index_done[i] = false;
        for(int k = 0; k < num; k++) {
            // 1 = comunicate with an element whose index is higher, -1 = the other way around
            int direction = (((base+k) / dist) % 2) * (-2) + 1;
            int line_dest = base+k + direction*dist;

            // Check each node singularly to verify if the operation is local or involves another process
            if(line_dest < base || line_dest >= base + num) {
                int P_dest = (line_dest <= amount_data_M-1) ? line_dest / M : (threshold + 1 + (line_dest-amount_data_M) / m);
                // Connect to an higher index process
                if(direction == 1) {
                    temp_data[2*req_count] = data[k];

                    snprintf(description, length_description, "Communicating: sending %d on level %d", base+k, d);
                    //hpmStart(2+k+d*(num+2), description);
                    MPI_Send(&data[k], sizeof(number), MPI_BYTE, P_dest, base+k, MPI_COMM_WORLD);
                    MPI_Irecv(&temp_data[2*req_count+1], sizeof(number), MPI_BYTE, P_dest, line_dest, MPI_COMM_WORLD, req+req_count);
                    //hpmStop(2+k+d*(num+2));
                }
                else {
                    temp_data[2*req_count+1] = data[k];

                    snprintf(description, length_description, "Communicating: sending %d on level %d", base+k, d);
                    //hpmStart(2+3*(k+d*num), description);
                    MPI_Send(&data[k], sizeof(number), MPI_BYTE, P_dest, base+k, MPI_COMM_WORLD);
                    MPI_Irecv(&temp_data[2*req_count], sizeof(number), MPI_BYTE, P_dest, line_dest, MPI_COMM_WORLD, req+req_count);
                    //hpmStop(2+3*(k+d*num));
                }
                indexes[req_count] = k;
                req_count++;
            }
            // Operation are betweeen elements stored in the same process, no need to comunicate
            else if(!index_done[k]) {
                int i = k;
                int j = k + direction*dist;
                number twiddle = pow(on, ((base+k) % dist) * d2);
                number a = data[i], b = data[j];

                if(direction == 1){
                    data[i] = a + b;
                    data[j] = twiddle * (a - b);
                }
                else{
                    data[j] = a + b;
                    data[i] = twiddle * (a - b);
                }

                index_done[i] = true;
                index_done[j] = true;
            }
        }

        snprintf(description, length_description, "Waiting at level %d", d);
        //hpmStart(2+num+d*(num+2), description);
        MPI_Waitall(req_count, req, MPI_STATUSES_IGNORE);
        //hpmStop(2+num+d*(num+2));

        // The number of send/received data is equal to the requests made
        for(int i = 0; i < req_count; i++) {
            int k = indexes[i];
            int direction = (((base+k) / dist) % 2) * (-2) + 1;
            number twiddle = pow(on, ((base+k) % dist) * d2);
            if(direction == 1)
                data[k] = temp_data[2*i] + temp_data[2*i+1];
            else
                data[k] = twiddle * (temp_data[2*i] - temp_data[2*i+1]);
        }

        snprintf(description, length_description, "Syncronization at level %d", d);
        //hpmStart(2+num+d*(num+2)+1, description);
        MPI_Barrier(MPI_COMM_WORLD);	// Can't start the next level if someone hasn't finished the current level
        //hpmStop(2+num+d*(num+2)+1);
    }

    delete[] temp_data;
    delete[] indexes;

    // Perform a bit reversal permutation
    vector<bool> swapped(m, false);
    req_count = 0;
    for(int k = 0; k < num; k++) {
        if(!swapped[k]) {
            int line = base + k;
            int br = bit_reversal(line, log2(n));
            int i = br - base;

            if(line == br) continue;	// No need to do anything as the element is already in the right position

            // The other element is in the same process, local swap
            if(base <= br && br < base + num) {
                number temp = data[k];
                data[k] = data[i];
                data[i] = temp;
                swapped[k] = true;
                swapped[i] = true;
            }
            // Have to exchange elements between processes
            else {
                int P_dest = (br <= amount_data_M-1) ? br / M : (threshold + 1 + (br-amount_data_M) / m);

                snprintf(description, length_description, "Permuting: sending %d", line);
                //hpmStart(levels*(num+2)+k, description);
                MPI_Send(data+k, sizeof(number), MPI_BYTE, P_dest, line, MPI_COMM_WORLD);
                MPI_Irecv(data+k, sizeof(number), MPI_BYTE, P_dest, br, MPI_COMM_WORLD, req+req_count);
                //hpmStop(levels*(num+2)+k);

                req_count++;
                swapped[k] = true;
            }
        }
    }
    //hpmStart(levels*(num+2)+num, "Waiting end of permutation");
    MPI_Waitall(req_count, req, MPI_STATUSES_IGNORE);
    //hpmStop(levels*(num+2)+num);
    delete[] req;

    //hpmStart(levels*(num+2)+num+1, "Syncronization before regrouping");
    MPI_Barrier(MPI_COMM_WORLD);
    //hpmStop(levels*(num+2)+num+1);

    // Return the whole vector in one process, P0
    //hpmStart(levels*(num+2)+num+2, "Regrouping data");
    MPI_Send(data, num * sizeof(number), MPI_BYTE, 0, rank, MPI_COMM_WORLD);
    if(rank == 0) {
        for(int k = 0; k <= threshold; k++)
            MPI_Recv(data + k * M, M * sizeof(number), MPI_BYTE, k, k, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for(int k = threshold+1; k < P; k++)
            MPI_Recv(data+amount_data_M+(k-threshold-1)*m, m * sizeof(number), MPI_BYTE, k, k, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    //hpmStop(levels*(num+2)+num+2);

    //hpmStop(1);

    //hpmTerminate(rank);
}

/*
 * Calcola il bit reversal di x, ovvero se in binario x = x0,x1,..,xn
 * allora il bit reversal è xn,..,x1,x0, il tutto assumedo che i bit
 * di x a cui siamo interessati siano i primi num_bit
*/
int bit_reversal(int x, int num_bit) {

    for(int k = 0; k < num_bit / 2; k++) {
        char bit_dx = (x >> k) & 1;	// Bit which is nearer to the least significant bit
        char bit_sx = (x >> (num_bit - 1 - k)) & 1;	// Bit which is nearer to the most significant bit
        char temp = bit_sx;

        if(bit_dx == 0)	// Set the bit_sx (k-th from the MSB) bit to the value stored in bit_dx
            x &= (~(1 << (num_bit - 1 - k)));
        else
            x |= (1 << (num_bit - 1 - k));

        if(temp == 0)
            x &= (~(1 << k));
        else
            x |= (1 << k);
    }

    return x;
}