
const int expulse_freq_arr[] = {290}; // f = 16e6/(freq_no+1)
const double expulse_duration_arr[] = {200}; // pulse width in microseconds
const double expulse_amp = 1.2;

const int NUM_AVERAGES = 10;
const int NUM_ECHOS = 80;   // number of times oscilloscope will be triggered during measurement

const int NUM_FREQ_ARRAY_ELEMENTS = sizeof(expulse_freq_arr)/sizeof(expulse_freq_arr[0]);
const int NUM_DURATION_ARRAY_ELEMENTS = sizeof(expulse_duration_arr)/sizeof(expulse_duration_arr[0]);

// sampling parameters
const double sample_freq = 200e3;
const double sample_amp  = 2;
// sample time after each echo (this value is for 90 degree pulse, 180 degree
// pulse is doubled) in microseconds
const long echo_sample_time = 16000; 

const int AD2_BUFFER_SIZE = double(echo_sample_time) * 2 * 1e-6 *  sample_freq;     // size of AD2 oscilloscope internal buffer [samples] (max: 8192)

const double ARDUINO_CLK_FREQ = 16e6;   // base clock frequency of arduino



// time between relay switch on and off and other operations (microseconds)
const long RELAY_DELAY = 200;

// time between trials of echos allowing magnetic spins to realign with the static field (ms)
const int TRIAL_DELAY = 7000;