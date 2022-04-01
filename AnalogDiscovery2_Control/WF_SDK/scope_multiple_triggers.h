//
// Created by campell on 2022-03-29.
//

/* include the necessary libraries */
#include <vector>
#include <string>

using namespace std;

/* include the constants and the WaveForms function library */
#ifdef _WIN32
#include "C:/Program Files (x86)/Digilent/WaveFormsSDK/inc/dwf.h"
#elif __APPLE__
#include "/Library/Frameworks/dwf.framework/Headers/dwf.h"
#else
#include <digilent/waveforms/dwf.h>
#endif

class scope_multiple_triggers {
private:
    class Trigger_Source {
        // trigger source names
    public:
        const TRIGSRC none = trigsrcNone;
        const TRIGSRC analog = trigsrcDetectorAnalogIn;
        const TRIGSRC digital = trigsrcDetectorDigitalIn;
        const TRIGSRC external[5] = {trigsrcNone, trigsrcExternal1, trigsrcExternal2, trigsrcExternal3, trigsrcExternal4};
    };

public:
    Trigger_Source trigger_source;
    void initialize(HDWF device_handle, int channel, double sampling_frequency, int buffer_size, double amplitude_range);
    void start_measurement(HDWF device_handle, int channel, int num_triggers, int buffer_size, double data_buffer[]);
    void save_data(HDWF device_handle, double buffer_data[], double time_data[], int total_num_samples, string file_name);
    void close(HDWF device_handle);
} scope_multiple_triggers;
