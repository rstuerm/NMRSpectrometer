/* WAVEFORM GENERATOR CONTROL FUNCTIONS: generate, close */

/* include the header */
#include "wavegen.h"

/* ----------------------------------------------------- */

void Wavegen::generate(HDWF device_handle, int channel, FUNC function, double offset, double frequency, double amplitude, double symmetry, double wait, double run_time, int repeat, vector<double> data) {
    /*
        generate an analog signal

        parameters: - device handle
                    - the selected wavegen channel (1-2)
                    - function - possible: custom, sine, square, triangle, noise, ds, pulse, trapezium, sine_power, ramp_up, ramp_down
                    - offset voltage in Volts
                    - frequency in Hz, default is 1KHz
                    - amplitude in Volts, default is 1V
                    - signal symmetry in percentage, default is 50%
                    - wait time in seconds, default is 0s
                    - run time in seconds, default is infinite (0)
                    - repeat count, default is infinite (0)
                    - data - list of voltages, used only if function=custom, default is empty
    */
    // enable channel
    channel--;
    FDwfAnalogOutNodeEnableSet(device_handle, channel, AnalogOutNodeCarrier, true);
    
    // set function type
    FDwfAnalogOutNodeFunctionSet(device_handle, channel, AnalogOutNodeCarrier, function);
    
    // load data if the function type is custom
    if (function == funcCustom) {
        FDwfAnalogOutNodeDataSet(device_handle, channel, AnalogOutNodeCarrier, data.data(), data.size());
    }
    
    // set frequency
    FDwfAnalogOutNodeFrequencySet(device_handle, channel, AnalogOutNodeCarrier, frequency);
    
    // set amplitude or DC voltage
    FDwfAnalogOutNodeAmplitudeSet(device_handle, channel, AnalogOutNodeCarrier, amplitude);
    
    // set offset
    FDwfAnalogOutNodeOffsetSet(device_handle, channel, AnalogOutNodeCarrier, offset);
    
    // set symmetry
    FDwfAnalogOutNodeSymmetrySet(device_handle, channel, AnalogOutNodeCarrier, symmetry);
    
    // set running time limit
    FDwfAnalogOutRunSet(device_handle, channel, run_time);
    
    // set wait time before start
    FDwfAnalogOutWaitSet(device_handle, channel, wait);
    
    // set number of repeating cycles
    FDwfAnalogOutRepeatSet(device_handle, channel, repeat);


    // Campbell Added
    // set PC as trigger source
    //FDwfAnalogOutTriggerSourceSet(device_handle, channel, trigsrcPC);

    // if channel selected is excitation pulse channel
    if (channel == 0){
        // set trigger source to external trigger 1
        FDwfAnalogOutTriggerSourceSet(device_handle, channel, trigsrcExternal1);
		FDwfAnalogOutRepeatTriggerSet(device_handle, channel, true);
    }
    
    // start
    FDwfAnalogOutConfigure(device_handle, channel, true);
    return;
}

/* ----------------------------------------------------- */

void Wavegen::close(HDWF device_handle) {
    /*
        reset the wavegen
    */
    FDwfAnalogOutReset(device_handle, -1);
    return;
}
