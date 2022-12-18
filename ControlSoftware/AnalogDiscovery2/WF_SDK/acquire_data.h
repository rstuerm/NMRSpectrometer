//
// Created by campell on 2022-03-08.
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

/* ----------------------------------------------------- */
/*
class scope_data2 {
public:
    vector<double> buffer;
    vector<double> time;
    scope_data2& operator=(const scope_data2&);
};

scope_data2& scope_data2::operator=(const scope_data2 &data) {
    if (this != &data) {
        buffer = data.buffer;
        time = data.time;
    }
    return *this;
}
*/

/* ----------------------------------------------------- */

class acquire_data {
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
    void initialize(HDWF device_handle, int channel, double sampling_frequency, double sampling_duration, double amplitude_range);
    void start_recording(HDWF device_handle, double data_buffer[], int channel, double sampling_frequency, double sampling_duration);
    void save_data(HDWF device_handle, double buffer_data[], double time_data[], int total_num_samples, string file_name);
    void close(HDWF device_handle);
} acquire_data;
