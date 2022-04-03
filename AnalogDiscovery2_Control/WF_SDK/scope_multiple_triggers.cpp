/*
 * adapted from /usr/share/digilent/waveforms/samples/c/analogin_trigger.cpp
 */

#include <iostream>
#include <fstream>
#include <string>
#include <unistd.h>
#include "scope_multiple_triggers.h"



// configure oscilloscope settings to perform multiple triggered acquisitions
void scope_multiple_triggers::initialize(HDWF device_handle, int channel, double sampling_frequency, int buffer_size, double amplitude_range) {

    // account for zero indexing of channels
    channel--;

    FDwfAnalogInChannelEnableSet(device_handle, channel, true);  // enable scope channel

    FDwfAnalogInFrequencySet(device_handle, sampling_frequency);       // set scope sampling frequency [Hz]
    FDwfAnalogInBufferSizeSet(device_handle, buffer_size);                  // max number of samples to take per acquisition
    FDwfAnalogInChannelRangeSet(device_handle, channel, amplitude_range);   // set scope input range [V]

    // set up trigger

    FDwfAnalogInTriggerAutoTimeoutSet(device_handle, 0);         // disable auto trigger
    FDwfAnalogInTriggerSourceSet(device_handle, trigsrcExternal2);  // set external trigger pin 2 as trigger source

	// FDwfAnalogInTriggerLevelSet(device_handle, 8.0);				// Set trigger level higher to avoid triggering on transmit pulse noise
	// FDwfAnalogInTriggerTypeSet(device_handle, trigtypePulse);
	// FDwfAnalogInTriggerLengthConditionSet(device_handle, triglenTimeout);
	// FDwfAnalogInTriggerLengthSet(device_handle, 4e-3);
	// FDwfAnalogInTriggerPositionSet(device_handle, double(buffer_size)/2.0/sampling_frequency);

	double voltage_min;
	double voltage_max;
	double voltage_steps;
	FDwfAnalogInChannelRangeInfo(device_handle, &voltage_min, &voltage_max, &voltage_steps);
	std::cout << "Min voltage: " << voltage_min << std::endl;
	std::cout << "Max voltage: " << voltage_max << std::endl;
	std::cout << "Voltage steps: " << voltage_steps << std::endl;

	FDwfAnalogInChannelRangeGet(device_handle, channel, &voltage_max);
	std::cout << "Voltage range: " << voltage_max << std::endl;

    int buffer_min;
    int buffer_max;
    FDwfAnalogInBufferSizeInfo(device_handle, &buffer_min, &buffer_max);
    std::cout << "Maximum buffer size: " << buffer_max << std::endl;
    std::cout << "Minimum buffer size: " << buffer_min << std::endl;

    // wait at least 2 seconds with Analog Discovery for the offset to stabilize, before the first reading after device open or offset/range change
    sleep(2);

    return;

}


void scope_multiple_triggers::start_measurement(HDWF device_handle, int channel, int num_triggers, int buffer_size, double data_buffer[], int enable_timeout, double timeout_time) {

    STS device_status;
	int sleep_time = 100;
	int timeout_max = timeout_time / (sleep_time * 1e-6);
	int timeout_counter = 0;

    // account for zero indexing of channels
    channel--;


    // start recording data (on trigger)
    FDwfAnalogInConfigure(device_handle, false, true);

    std::cout << "Starting repeated acquisitions:" << std::endl;

    // loop for each expected trigger pulse from Arduino
    for (int i = 0; i < num_triggers; i++){

		timeout_counter = 0;

        // poll AD2 until acquisition is complete
        while(true){
            FDwfAnalogInStatus(device_handle, true, &device_status);
            if(device_status == DwfStateDone){
                break;
            }
            usleep(sleep_time);

			if (enable_timeout == 1){
                timeout_counter++;
			}

			if (timeout_counter >= timeout_max){
				std::cout << "Error: Device Timeout" << std::endl;
				break;
			}
        }

		if (timeout_counter >= timeout_max){
			break;
		}

        // transfer data from AD2 buffer to array in PC
        FDwfAnalogInStatusData(device_handle, channel, &data_buffer[i*buffer_size], buffer_size);
        // &data_buffer[i*buffer_size]
    }

    std::cout << "data_buffer[0] " << data_buffer[0] << std::endl;

    return;
}



// save measured time-domain data in .csv file for further processing
// copied from WaveForms-SDK-Getting-Started-Cpp/test_scope-wavegen.cpp
void scope_multiple_triggers::save_data(HDWF device_handle, double buffer_data[], double time_data[], int total_num_samples, string file_name){

    // save data
    std::ofstream output_file;
    output_file.open(file_name);

    output_file << "time [ms], voltage [V]\n";
    for (int index = 0; index < total_num_samples; index++){
        output_file << to_string(time_data[index]) + "," + to_string(buffer_data[index]) + "\n";
    }

    output_file.close();
}



// reset oscilloscope settings
void scope_multiple_triggers::close(HDWF device_handle){
    FDwfAnalogInReset(device_handle);
    return;
}

