
const int expulse_freq_arr[] = {290, 305}; // f = 16e6/(freq_no+1)
const double expulse_duration_arr[] = {200, 190}; // pulse width in microseconds
const double expulse_amp = 1.2;

const int NUM_AVERAGES = 2;
const int NUM_ECHOS = 10;

const int NUM_FREQ_ARRAY_ELEMENTS = sizeof(expulse_freq_arr)/sizeof(expulse_freq_arr[0]);
const int NUM_DURATION_ARRAY_ELEMENTS = sizeof(expulse_duration_arr)/sizeof(expulse_duration_arr[0]);


// sampling parameters
const double sample_freq = 500e3;
const double sample_time = 500e-3;
const double sample_amp  = 2;

// sample time after each echo (for 90 degree pulse, 180 degree pulse is doubled) in microseconds
const long echo_sample_time = 16000; 

// time between relay switch on and off and other operations in microseconds
const long RELAY_DELAY = 200;

// time between trials of echos allowing magnetic spins ro realign with the static field
const int TRIAL_DELAY = 20000;