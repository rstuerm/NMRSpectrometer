/*
 * adapted from /usr/share/digilent/waveforms/samples/c/analogin_record.cpp
 */

#include <iostream>
#include <fstream>
#include <string>
#include "acquire_data.h"

void acquire_data::initialize(HDWF device_handle, int channel, double sampling_frequency, double sampling_duration, double amplitude_range) {

    // adjust value since channel is zero indexed
    channel--;

    FDwfAnalogInChannelEnableSet(device_handle, channel, true);

    FDwfAnalogInChannelRangeSet(device_handle, channel, amplitude_range);

    // set oscilloscope to record mode which allows longer sampling times not limited by AD2 internal buffer size
    FDwfAnalogInAcquisitionModeSet(device_handle, acqmodeRecord);
    FDwfAnalogInFrequencySet(device_handle, sampling_frequency);    // set sample frequency
    FDwfAnalogInRecordLengthSet(device_handle, sampling_duration);     // set duration of measurement

    // set trigger source to external trigger pin 2
    FDwfAnalogInTriggerAutoTimeoutSet(device_handle, 0);  // disable auto trigger
    FDwfAnalogInTriggerSourceSet(device_handle, trigsrcExternal2);

    // wait at least 2 seconds with Analog Discovery for the offset to stabilize, before the first reading after device open or offset/range change
    sleep(2);

    return;
}



void acquire_data::start_recording(HDWF device_handle, double data_buffer[], int channel, double sampling_frequency, double sampling_duration){

    int current_num_samples = 0;   // number of samples currently read from AD2
    int total_num_samples = (int)(sampling_frequency * sampling_duration);  // total number of samples expected during measurement
    STS device_status;
    int num_samples_available;   // number of samples available in AD2 memory
    int num_samples_lost;        // number of samples lost (overwritten) since last transfer
    int num_samples_corrupted;   // number of samples corrupted during transfer
    bool data_lost = false;      // flag to indicate if data was lost during measurement
    bool data_corrupted = false; // flag to indicate if data was corrupted during measurement


    // adjust value since channels are zero indexed
    channel--;


    // start recording data (on trigger)
    FDwfAnalogInConfigure(device_handle, false, true);

    std::cout << "Recording..." << std::endl;

    // poll AD2 until expected number of data points received
    while (current_num_samples < total_num_samples){
        // get status of AD2
        if(!FDwfAnalogInStatus(device_handle, true, &device_status)){
            std::cout << "error" << std::endl;
            break;
        }

        if (current_num_samples == 0 && (device_status == stsCfg || device_status == stsPrefill || device_status == stsArm)){
            // Acquisition not yet started
            continue;
        }

        // get number of available samples in AD2 (also retrieves number of samples lost or corrupted since last transfer)
        FDwfAnalogInStatusRecord(device_handle, &num_samples_available, &num_samples_lost, &num_samples_corrupted);

        // take into account any samples lost between transfers
        current_num_samples += num_samples_lost;

        if(num_samples_lost){
            data_lost = true;
        }
        if (num_samples_corrupted){
            data_corrupted = true;
        }
        if (!num_samples_available){
            continue;
        }

        // get samples and save them into data_buffer
        FDwfAnalogInStatusData(device_handle, channel, &data_buffer[current_num_samples], num_samples_available);
        current_num_samples += num_samples_available;
    }

    std::cout << "done" << std::endl;

    std::cout << "Current_num_samples: " << current_num_samples << std::endl;
    std::cout << "Total_num_samples: " << total_num_samples << std::endl;

    if(data_lost){
        std::cout << "Samples were lost! Reduce frequency" << std::endl;
    }
    else if (data_corrupted){
        std::cout << "Samples could be corrupted! Reduce frequency" << std::endl;
    }


    return;

}


// save measured time-domain data in .csv file for further processing
// copied from WaveForms-SDK-Getting-Started-Cpp/test_scope-wavegen.cpp
void acquire_data::save_data(HDWF device_handle, double buffer_data[], double time_data[], int total_num_samples, string file_name){

    // save data
    std::ofstream output_file;
    //output_file.open("test-output-data.csv");
    output_file.open(file_name);
    output_file << "time [ms], voltage [V]\n";
    for (int index = 0; index < total_num_samples; index++){
        output_file << to_string(time_data[index]) + "," + to_string(buffer_data[index]) + "\n";
    }
    output_file.close();
}

// reset oscilloscope settings
void acquire_data::close(HDWF device_handle){
    FDwfAnalogInReset(device_handle);
    return;
}
