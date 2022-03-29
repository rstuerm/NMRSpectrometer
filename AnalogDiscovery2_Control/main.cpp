/*
 *  Sources:
 *      - https://github.com/Digilent/WaveForms-SDK-Getting-Started-Cpp
 *
 */

// Note: The below include code was taken from /usr/share/digilent/waveforms/samples/c/sample.h
// is it actually needed? I have no idea

#include <stdio.h>
#include <stdlib.h>
#include <string>

// Includes consistent parameters as Arduino file
#include "../Arduino_Control/src/Parameters.h"

#ifdef WIN32
#include "../../inc/dwf.h"
#elif __APPLE__
#include "/Library/Frameworks/dwf.framework/Headers/dwf.h"
#else
#include <digilent/waveforms/dwf.h>
#endif

#ifdef WIN32
#include <windows.h>
    #define Wait(ts) Sleep((int)(1000*ts))
#else
#include <unistd.h>
#include <sys/time.h>
#define Wait(ts) usleep((int)(1000000*ts))
#endif

#include <iostream>
#include "WF_SDK/WF_SDK.h"
#define EXTRA_ELEMENTS 1000

// function prototypes
int take_NMR_measurement(device_data this_device, double data_buffer[], double time_data[], int total_num_samples, double expulse_freq, double expulse_amp, double expulse_duration, double sample_freq, double sample_time, double sample_amp, string file_name);
void clear_buffer(double buffer[], int num_elements);


int main(){

    // output file name parameters
	string file_name_location = "../Processing/";  // file location, not sure about windows compatibility
    string file_name_base = "test_data_";          // file prefix
    string file_extension = ".csv";                // file type of output file
    string file_name;                              // full output file name

	double expulse_duration;
	double expulse_freq;

    /* ----------------------------------------------------- */

    // connect to the device
    device_data this_device;
    this_device = device.open();

    // check for connection errors
    device.check_error(this_device.handle);

    /* ----------------------------------------------------- */

    // turn on power supplies
    supplies.switch_variable(this_device.handle, true, true, false, 3.3, 0);

    // generate test output waveform, feed to input of oscilloscope
    wavegen.generate(this_device.handle, 2, wavegen.function.sine, 0, 130e3, 1, 50, 0, 0, 0);


    int total_num_samples = (int)(sample_freq * sample_time);  // total number of samples expected during measurement
    double data_buffer[total_num_samples + EXTRA_ELEMENTS];    // array to hold oscilloscope data
    // note: AD2 record mode doesn't return exactly total_num_samples data points, typically adds ~40 more

    // calculate acquisition time [ms]
    double time_data[total_num_samples + EXTRA_ELEMENTS];
    for (int index = 0; index < total_num_samples; index++){
        time_data[index] = (index * 1e3) / sample_freq;
    }


    // -------- Start main measurement loop -------- //


    // sweep through excitation pulse frequencies
    for (int i = 0; i < NUM_FREQ_ARRAY_ELEMENTS; i++){
        // sweep through excitation pulse durations
        for (int j = 0; j < NUM_DURATION_ARRAY_ELEMENTS; j++){
			// sweep through averages of the same pulse parameters
			for (int k = 0; k < NUM_AVERAGES; k++){

				std::cout << "Beginning measurement: " << " i=" << i << " j=" << j << std::endl;

				expulse_freq     = 16e6/(double(expulse_freq_arr[i])+1);
				expulse_duration = double(1/expulse_freq)*0.99;

				std::cout << "Excitation pulse frequency: " << expulse_freq << std::endl;
				std::cout << "Excitation pulse duration: " << expulse_duration << std::endl;

				// set all values in data buffer to 0
				clear_buffer(data_buffer, total_num_samples + EXTRA_ELEMENTS);

				// create full name of output file
				file_name = file_name_location + file_name_base + to_string(i) + "_" + to_string(j) + "_" + to_string(k) + file_extension;

				// take NMR spectrum measurement
				take_NMR_measurement(this_device, data_buffer, time_data, total_num_samples, expulse_freq, expulse_amp, expulse_duration, sample_freq, sample_time, sample_amp, file_name);

				sleep(2);

			}
        }
    }

    /*
    for (int index = 0; index < 3; index++){

        // set all values in data buffer to 0
        clear_buffer(data_buffer, total_num_samples + EXTRA_ELEMENTS);

        // create full name of output file
        file_name = file_name_base + to_string(index) + file_extension;

        // take NMR spectrum measurement
        take_NMR_measurement(this_device, data_buffer, time_data, total_num_samples, expulse_freq, expulse_amp, expulse_duration, sample_freq, sample_time, sample_amp, file_name);

        sleep(2);
    }
     */


    /* ----------------------------------------------------- */

    // reset the wavegen
    wavegen.close(this_device.handle);

    // reset oscilloscope
    acquire_data.close(this_device.handle);

    // reset power supplies
    supplies.close(this_device.handle);

    // close the connection with AD2
    device.close(this_device.handle);

    return 0;
}



int take_NMR_measurement(device_data this_device, double data_buffer[], double time_data[], int total_num_samples, double expulse_freq, double expulse_amp, double expulse_duration, double sample_freq, double sample_time, double sample_amp, string file_name) {

    std::cout << "Beginning Measurement..." << std::endl;

    // generate excitation pulse
    wavegen.generate(this_device.handle, 1, wavegen.function.sine, 0, expulse_freq, expulse_amp, 50, 0, expulse_duration, 0);
    std::cout << "Excitation pulse armed..." << std::endl;


    // measure NMR signal
    acquire_data.initialize(this_device.handle, 2, sample_freq, sample_time, sample_amp);
    std::cout << "Oscilloscope armed..." << std::endl;
    acquire_data.start_recording(this_device.handle, data_buffer, 2, sample_freq, sample_time);


    // save recorded data into .csv
    std::cout << "Writing data to .csv" << std::endl;
    acquire_data.save_data(this_device.handle, data_buffer, time_data, total_num_samples, file_name);


    /* ----------------------------------------------------- */


    //std::cout << "\nPress Enter to exit...";
    //cin.get();

    // reset the wavegen
    //wavegen.close(this_device.handle);

    // reset oscilloscope
    //acquire_data.close(this_device.handle);

    return 0;
}


// set all elements in a given buffer to 0
void clear_buffer(double buffer[], int num_elements){

    for (int i = 0; i < num_elements; i++){
        buffer[i] = 0;
    }
    return;
}
