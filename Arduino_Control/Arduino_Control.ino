#include "src/Parameters.h"

#define TRANSMIT_TRIGGER 12
#define OSCILLOSCOPE_TRIGGER 11

#define TRANSMIT_PIN 10
#define RECEIVE_PIN 9

int serial_control = 0;

void setup() 
{

	Serial.begin(19200);

	pinMode(LED_BUILTIN, OUTPUT);
	digitalWrite(LED_BUILTIN, LOW);

	pinMode(TRANSMIT_PIN, OUTPUT);
	digitalWrite(TRANSMIT_PIN, LOW);

	pinMode(RECEIVE_PIN, OUTPUT);
	digitalWrite(RECEIVE_PIN, LOW);	

	pinMode(OSCILLOSCOPE_TRIGGER, OUTPUT);
	digitalWrite(OSCILLOSCOPE_TRIGGER, LOW);	

	pinMode(TRANSMIT_TRIGGER, OUTPUT);
	digitalWrite(TRANSMIT_TRIGGER, LOW);	

	delay(1000);
	
}

void loop() 
{

	if (Serial.available() > 0) 
	{
		serial_control = Serial.parseInt();
	}

	// Send integer 1 to device over serial to start sequence
	if (serial_control == 1)
	{
		// delay(TRIAL_DELAY);
		for (int i = 0; i < NUM_FREQ_ARRAY_ELEMENTS; i++)
		{
			for (int j = 0; j < NUM_DURATION_ARRAY_ELEMENTS; j++)
			{

				for (int k = 0; k < NUM_AVERAGES; k++)
				{
					Serial.print(i);
					Serial.print(" ");
					Serial.print(j);
					Serial.print(" ");
					Serial.println(k);

					Pulse(expulse_freq_arr[i], expulse_duration_arr[j], echo_delay_time, 0);

					for (int l = 0; l < NUM_ECHOS + 5; l++)
					{
						Pulse(expulse_freq_arr[i], expulse_duration_arr[j]*2, echo_delay_time*2, 1);
					}

					delay(TRIAL_DELAY);

				}
			}
		}
		serial_control = 0;
	}

	digitalWrite(LED_BUILTIN, HIGH);
	delay(500);
	digitalWrite(LED_BUILTIN, LOW);
	delay(500);

}

void Pulse(int freq, int length, long echo_sample_time, int sample_flag)
{

	// Enable transmit relay and wait for relay to stabilize
	digitalWrite(TRANSMIT_PIN, HIGH);
	delayMicroseconds(RELAY_DELAY);

	// Generate transmit pulse
	TCCR1A = (1 << COM1B1) | (1 << COM1B0) | (1 << WGM11);
	TCCR1B = (1 << WGM13) | (1 << WGM12);
	ICR1 = freq;
	OCR1B = freq/2; 
	TCCR1B |= (1 << CS10);

	delayMicroseconds(length); 
	digitalWrite(TRANSMIT_TRIGGER, LOW);

	/* Wait for RF coil voltage to ringdown before turning of transmit relay */
	delayMicroseconds(RELAY_DELAY);
	digitalWrite(TRANSMIT_PIN, LOW);

	// Small delay before enabling receive relay
	delayMicroseconds(RELAY_DELAY); 
	digitalWrite(RECEIVE_PIN, HIGH);
	delayMicroseconds(RELAY_DELAY*2); 

	/* Only trigger sample if flag set, allows first 90 degree pulse to not be
	sampled */
	if (sample_flag)
	{
		/* Start sampling at half of echo sample time delay since sampling
		trigger is center point of received data */
		arbDelayMicroseconds(echo_sample_time/2);
		digitalWrite(OSCILLOSCOPE_TRIGGER, HIGH);
		arbDelayMicroseconds(echo_sample_time/2);
		digitalWrite(OSCILLOSCOPE_TRIGGER, LOW);

		/* Account for double the relay delay to ensure symmetric 180 degree
		pulse compared to 90 degree pulse */
		delayMicroseconds(RELAY_DELAY*3);
	}
	else
	{
		arbDelayMicroseconds(echo_sample_time);
	}

	digitalWrite(RECEIVE_PIN, LOW);

}


// Delay function which allows longer than maximum possible delay
void arbDelayMicroseconds(long delay_time)
{
	const long MAX_ARDUINO_DELAY_TIME = 16383;

	for(int i = 0; i < delay_time / MAX_ARDUINO_DELAY_TIME; i++)
	{
		delayMicroseconds(MAX_ARDUINO_DELAY_TIME);
	}
	delayMicroseconds(delay_time % MAX_ARDUINO_DELAY_TIME);

}