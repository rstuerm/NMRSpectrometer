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
int take_NMR_measurement(device_data this_device, double data_buffer[], double time_data[], int total_num_samples, double expulse_freq, double expulse_duration, string file_name, int enable_timeout);
void clear_buffer(double buffer[], int num_elements);


int main(){

    // output file name parameters
	string file_name_location = "../Processing/";  // output file location
    // string file_name_location = "";
    string file_name_base = "test_data_";          // file prefix
    string file_extension = ".csv";                // output file extension
    string file_name;                              // full output file name

	double expulse_duration;    // period of excitation pulse
	double expulse_freq;        // frequency of excitation pulse


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


    int total_num_samples = (int)(NUM_ECHOS * AD2_BUFFER_SIZE);  // total number of samples expected during measurement
    double data_buffer[total_num_samples + EXTRA_ELEMENTS];    // array to hold oscilloscope data
    // note: AD2 record mode doesn't return exactly total_num_samples data points, typically adds ~40 more

    // calculate acquisition time [ms]
    double time_data[total_num_samples + EXTRA_ELEMENTS];
    for (int index = 0; index < total_num_samples; index++){
        time_data[index] = (index * 1e3) / sample_freq;
    }


    // -------- Start main measurement loop -------- //

    int enable_timeout = 0;

    // sweep through excitation pulse frequencies
    for (int i = 0; i < NUM_FREQ_ARRAY_ELEMENTS; i++){
        // sweep through excitation pulse durations
        for (int j = 0; j < NUM_DURATION_ARRAY_ELEMENTS; j++){
			// sweep through averages of the same pulse parameters
			for (int k = 0; k < NUM_AVERAGES; k++){

				std::cout << "Beginning measurement: " << " i=" << i << " j=" << j << " k=" << k << std::endl;

				expulse_freq     = ARDUINO_CLK_FREQ / (double(expulse_freq_arr[i]) + 1);  // calculate excitation pulse frequency
                // note: expulse_freq is an integer division of the Arduino base clock frequency (16 MHz)
				expulse_duration = double(1/expulse_freq) * 0.99;   // calculate period of excitation pulse
                // note: Arduino forms longer excitation pulses by chaining single periods of the excitation pulse
                // ex. a 100 us long 100 kHz pulse is generated by sending 100 triggers

				std::cout << "Excitation pulse frequency: " << expulse_freq << std::endl;
				std::cout << "Excitation pulse duration: " << expulse_duration << std::endl;

				// set all values in data buffer to 0
				clear_buffer(data_buffer, total_num_samples + EXTRA_ELEMENTS);

				// create full name of output file
				file_name = file_name_location + file_name_base + to_string(i) + "_" + to_string(j) + "_" + to_string(k) + file_extension;

				if (i == 0 && j == 0 && k == 0){
					enable_timeout = 0;
				}
				else {
					enable_timeout = 1;
				}

				// take NMR spectrum measurement
				take_NMR_measurement(this_device, data_buffer, time_data, total_num_samples, expulse_freq, expulse_duration, file_name, enable_timeout);

                // wait a short duration between each measurement
				sleep(2);

                std::cout << "===========================\n" << std::endl;

			}
        }
    }


    /* ----------------------------------------------------- */

    // reset the wavegen
    wavegen.close(this_device.handle);

    // reset oscilloscope
    scope_multiple_triggers.close(this_device.handle);

    // reset power supplies
    supplies.close(this_device.handle);

    // close the connection with AD2
    device.close(this_device.handle);

    return 0;
}


int take_NMR_measurement(device_data this_device, double data_buffer[], double time_data[], int total_num_samples, double expulse_freq, double expulse_duration, string file_name, int enable_timeout) {

    std::cout << "Beginning Measurement..." << std::endl;

    // generate excitation pulse, transmitted on external trigger signal
    wavegen.generate(this_device.handle, 1, wavegen.function.sine, 0, expulse_freq, expulse_amp, 50, 0, expulse_duration, 0);
    std::cout << "Excitation pulse armed..." << std::endl;


    // measure NMR signal
    //acquire_data.initialize(this_device.handle, 2, sample_freq, sample_time, sample_amp);
    //acquire_data.start_recording(this_device.handle, data_buffer, 2, sample_freq, sample_time);
    scope_multiple_triggers.initialize(this_device.handle, 2, sample_freq, AD2_BUFFER_SIZE, sample_amp);
    std::cout << "Oscilloscope armed..." << std::endl;
    scope_multiple_triggers.start_measurement(this_device.handle, 2, NUM_ECHOS, AD2_BUFFER_SIZE, data_buffer, enable_timeout, TRIAL_DELAY/1e3*1.5);


    std::cout << "data_buffer[0] " << data_buffer[0] << std::endl;

    // save recorded data into .csv
    std::cout << "Writing data to .csv" << std::endl;
    scope_multiple_triggers.save_data(this_device.handle, data_buffer, time_data, total_num_samples, file_name);


    /* ----------------------------------------------------- */


    //std::cout << "\nPress Enter to exit...";
    //cin.get();

    // reset the wavegen
    //wavegen.close(this_device.handle);

    // reset oscilloscope
    //scope_multiple_triggers.close(this_device.handle);

    return 0;
}


// set all elements in a given buffer to 0
void clear_buffer(double buffer[], int num_elements){

    for (int i = 0; i < num_elements; i++){
        buffer[i] = 0;
    }
    return;
}
