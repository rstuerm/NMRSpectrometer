# %%

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

path = os.path.dirname(os.path.abspath(__file__))

CAPTURE_DURATION = 0.98
START_TIME = 0
REFERENCE_START_TIME = 1

NUM_FREQ_ARRAY_ELEMENTS = 1
NUM_DURATION_ARRAY_ELEMENTS = 1
NUM_AVERAGES = 3

for i in list(range(NUM_FREQ_ARRAY_ELEMENTS)):

	for j in list(range(NUM_DURATION_ARRAY_ELEMENTS)):

		signal_dBV_ave = 0
		reference_dBV_ave = 0

		print(str(i) + " " + str(j) + "\n")

		for k in list(range(NUM_AVERAGES)):

			data = np.loadtxt(path + '/test_data_' + str(i) + '_' + str(j) + '_' + str(k) + '.csv', delimiter=',', unpack=True, skiprows=1)[0:2]

			t = data[0]*1e-3
			sample_rate = 1/(t[1]-t[0])
			signal = data[1]

			if np.all(signal == 0):
				NUM_AVERAGES = NUM_AVERAGES - 1
				continue

			signal_trimmed = (signal)[(t <= START_TIME + CAPTURE_DURATION) & (t > START_TIME)]
			reference_trimmed = (signal)[(t <= REFERENCE_START_TIME + CAPTURE_DURATION) & (t > REFERENCE_START_TIME)]

			f = np.fft.fftfreq(len(signal_trimmed),1/sample_rate) 

			signal_amplitude = 2/(sample_rate*CAPTURE_DURATION)*abs(np.fft.fft(signal_trimmed))
			reference_amplitude = 2/(sample_rate*CAPTURE_DURATION)*abs(np.fft.fft(reference_trimmed))

			signal_dBV = 20*np.log10(signal_amplitude/1)
			reference_dBV = 20*np.log10(reference_amplitude/1)

			signal_dBV_ave += signal_dBV
			reference_dBV_ave += reference_dBV

		signal_dBV_ave = signal_dBV_ave/NUM_AVERAGES
		reference_dBV_ave = reference_dBV_ave/NUM_AVERAGES

		min_f = 50e3
		max_f = 60e3

		f_trimmed = f[(f > min_f) & (f < max_f)]
		signal_dBV = signal_dBV[(f > min_f) & (f < max_f)]
		reference_dBV = reference_dBV[(f > min_f) & (f < max_f)]

		signal_peak = np.max(signal_dBV)
		noise_ave = np.sum(signal_dBV)/np.size(f_trimmed)
		print(noise_ave)
		print(signal_peak)
		print((f_trimmed[signal_dBV >= signal_peak])/1e3)
		print(signal_peak-noise_ave)
		print("\n")

		plt.close('all')

		rc = {"font.family" : "serif", "mathtext.fontset" : "stix"}
		plt.rcParams.update(rc)
		plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
		plt.rcParams['font.size'] = 10
		mpl.rcParams['axes.linewidth'] = 1

		fig, ax = plt.subplots()
		fig.subplots_adjust(left=0.15, bottom=0.16, right=0.85, top=0.97)

		width  = 3.487
		height = width / 1.618
		fig.set_size_inches(width, height)


		ax.xaxis.set_tick_params(which='major', size=5, width=1, direction='in')
		ax.xaxis.set_tick_params(which='minor', size=3, width=1, direction='in')

		ax.yaxis.set_tick_params(which='major', size=5, width=1, direction='in')
		ax.yaxis.set_tick_params(which='minor', size=3, width=1, direction='in')


		step_y1 = 25
		step_x = 5


		# ax.set_xlim(min_f/1e3-step_x/4, max_f/1e3+step_x/4)

		# ax.set_ylim(-112.5-step_y1/4, -37.5+step_y1/4)

		ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(step_x))
		ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(step_x/2))

		ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(step_y1))
		ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(step_y1/2))


		ax.set_xlabel('$f \\quad (\\rm{kHz})$', labelpad=1)
		ax.set_ylabel('$\\rm{Amplitude} \\quad (\\rm{dBV})$', labelpad=4)

		mpl.rcParams['axes.unicode_minus'] = False # uses dash instead of minus for axis numbers


		ax.plot(f_trimmed/1e3, signal_dBV, linewidth=0.5, color='r', linestyle='solid',markersize='2', label='Input')
		ax.plot(f_trimmed/1e3, reference_dBV, linewidth=0.5, color='k', linestyle='solid',markersize='2', label='Input')

		plt.savefig(path + '/FFT_' + str(i) + '_' + str(j)  + '.pdf')



		plt.close('all')

		rc = {"font.family" : "serif", "mathtext.fontset" : "stix"}
		plt.rcParams.update(rc)
		plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
		plt.rcParams['font.size'] = 10
		mpl.rcParams['axes.linewidth'] = 1

		fig, ax = plt.subplots()
		fig.subplots_adjust(left=0.15, bottom=0.16, right=0.85, top=0.97)

		width  = 3.487
		height = width / 1.618
		fig.set_size_inches(width, height)


		ax.xaxis.set_tick_params(which='major', size=5, width=1, direction='in')
		ax.xaxis.set_tick_params(which='minor', size=3, width=1, direction='in')

		ax.yaxis.set_tick_params(which='major', size=5, width=1, direction='in')
		ax.yaxis.set_tick_params(which='minor', size=3, width=1, direction='in')


		step_y1 = 25
		step_x = 5

		# ax.set_xlim(0, 600)

		# ax.set_ylim(-112.5-step_y1/4, -37.5+step_y1/4)

		# ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(step_x))
		# ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(step_x/2))

		# ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(step_y1))
		# ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(step_y1/2))


		ax.set_xlabel('$t \\quad (\\mu\\rm{s})$', labelpad=1)
		ax.set_ylabel('$\\rm{Amplitude} \\quad (\\rm{mV})$', labelpad=4)

		mpl.rcParams['axes.unicode_minus'] = False # uses dash instead of minus for axis numbers

		ax.plot(t/1e-6, signal/1e-3, linewidth=1, color='k', linestyle='solid',markersize='2', label='Input')

		plt.savefig(path + '/Time_' + str(i) + '_' + str(j)  + '.pdf')


