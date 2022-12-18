from paperplotter import plot

import numpy as np
import scipy.special as sp
from scipy import integrate
from scipy.misc import derivative

import os
path = os.path.dirname(os.path.abspath(__file__))

# For quantization of values
def round_nearest(x, a):
	return np.round(x / a) * a

# Constants 
mu = 4*np.pi*1e-7 # permeability of free space in N/A^2
gamma = 267.522e6 # gyromagnetic ratio of hydrogen in rad/s/T
T2 = 2 # water spin relaxation time estimate in s
h = 6.62607004e-34 # Planck constant
kb = 1.38064852e-23 # Boltzmann constant
T = 305 # ~30 C in coil with shielding but no cooling and open top
Ns = 110.4*6.022e23/0.001 # proton density in water as protons/m^3 (110.4 M -> Mol/L, 1 L = 0.001 m^3)

I_RF_Solenoid = 72e-3/2/2 # Measured RF current amplitude (half of peak), halved again to account for only one polarization direction
Q_Solenoid = 7.9 # measured quality factor of solenoid RF coil
transmit_width_Solenoid = 200e-6
I_RF_Helmholtz = 17e-3/2/2
Q_Helmholtz = 5.9 # measured quality factor of Helmholtz RF coil
gain = 50**2 # RF receive circuit gain

l = 2e-2 # edge length of cube for sample in m
r = np.linspace(0, l/2, 1001) # radius to evaluate at in m

# function to calculate the magnetic field at the point r,z inside a helmholtz coil 
def helmholtzB(r,z,R,N,I):

	k_1 = 4 * R * r / ( (R+r)**2 + (z+R/2)**2 )
	k_2 = 4 * R * r / ( (R+r)**2 + (z-R/2)**2 )

	Bz = (mu/(2*np.pi) * N*I) * ( (R+r)**2 + (z+R/2)**2 )**(-1/2) * ( sp.ellipk(k_1**2) + ( R**2 - r**2 - (z+R/2)**2 ) / ( (R-r)**2 + (z+R/2)**2 ) * sp.ellipe(k_1**2) )

	Bz += (mu/(2*np.pi) * N*I) * ( (R+r)**2 + (z-R/2)**2 )**(-1/2) * ( sp.ellipk(k_2**2) + ( R**2 - r**2 - (z-R/2)**2 ) / ( (R-r)**2 + (z-R/2)**2 ) * sp.ellipe(k_2**2) )

	return Bz

# function to calculate the magnetic field at the point r,z inside a solenoid 
def solenoidB(r,z,R,L,n,I):

	xi_pos = z + L/2
	xi_neg = z - L/2

	phi_pos = np.arctan(np.abs(xi_pos/(R-r)))
	phi_neg = np.arctan(np.abs(xi_neg/(R-r)))
	k_pos = np.sqrt(4*R*r/(xi_pos**2+(R+r)**2))
	k_neg = np.sqrt(4*R*r/(xi_neg**2+(R+r)**2))

	k_prime_pos = np.sqrt(1-k_pos**2)
	k_prime_neg = np.sqrt(1-k_neg**2)

	m_pos = (1-k_prime_pos)/(1+k_prime_pos)
	m_neg = (1-k_prime_neg)/(1+k_prime_neg)

	Bz = mu*n*I/4 * (xi_pos/np.abs(xi_pos)) * ( m_pos*(1+2*k_prime_pos)*2*phi_pos/np.pi + (m_pos**2*(1-m_pos)/4+2*k_prime_pos)*np.sin(phi_pos) )
	Bz -= mu*n*I/4 * (xi_neg/np.abs(xi_neg)) * ( m_neg*(1+2*k_prime_neg)*2*phi_neg/np.pi + (m_neg**2*(1-m_neg)/4+2*k_prime_neg)*np.sin(phi_neg) )
	# Magnetic field in T (N A^-1 m^-1)

	return Bz


# Various polarization and RF coil designs with fabrication parameters included 

def PolarizationCoil_Implemented(r,z):

	R = 115e-3/2 # radius from outer diameter of 4" pipe for polarizer coil in m
	L = 300e-3 # length of polarization coil in m
	gauge = 0.04*0.0254 # 18 AWG wire diameter in m
	R_shunt = 1.28 # shunt resistor for current measurement in ohms
	V_meas = 1.43 # voltage measured across shunt resistor in volts
	I = V_meas/R_shunt # meausured current going through the polarization coil
	shielding_loss = 1.07/1.14 # measured decrease in field strength due to shielding
	Bz = solenoidB(r,z,R,L,1/gauge,I)
	return Bz*shielding_loss

def PolarizationCoil_Shim(r,z):

	R = 115e-3/2 # radius from outer diameter of 4" pipe for polarizer coil in m
	L = 300e-3 # length of polarization coil in m
	gauge = 0.04*0.0254 # 18 AWG wire diameter in m
	I = 2.4 # current supplied to polarization coil
	I_shim = 2 # current supplied to shim coil
	N_shim = 10 # number of turns of wires in shim coil (per side)
	Bz = 0
	Bz += helmholtzB(r,z,R+gauge,N_shim,I_shim)
	Bz += solenoidB(r,z,R,L,1/gauge,I)
	return Bz

def PolarizationCoil_Michal(r,z):
	
	R = 115e-3/2 # radius from outer diameter of 4" pipe for polarization coil in m
	L = 150e-3 # length of polarizer coil in m
	gauge = 0.04*0.0254 # 18 AWG wire diameter in m
	I = 3 # current supplied to polarization coil
	Bz = solenoidB(r,z,R,L,1/gauge,I)
	return Bz

def RFCoil_Solenoid(r,z,I):

	N = 56*6 # number of turns of wires (6 layers with 56 turns per layer)
	R = 2.75*0.0254/2 # radius from outer diameter of 2.75" pipe for RF coil in m
	L = 3*0.0254 # length of polarizer coil in m (3 in)
	Bz = solenoidB(r,z,R,L,N/L,I)
	return Bz

def RFCoil_Helmholtz(r,z,I):

	N = 150 # number of turns of wire
	R = 2*0.0254/2 # radius from outer diameter of 2.75" pipe for RF coil in m
	Bz = helmholtzB(r,z,R,N,I) 
	return Bz


def StaticField_ave(func_Staticfield):

	# Cube for polarization coil using transformation of r to x and y
	def start(x,y,z):
		return func_Staticfield(np.sqrt(x**2+y**2),z)

	def int1(y,z):
		return integrate.quad(start, -l/2, l/2, epsabs=1.49e-3, epsrel=1.49e-3,  args=(y,z))[0]

	def int2(z):
		return integrate.quad(int1, -l/2, l/2, epsabs=1.49e-3, epsrel=1.49e-3,  args=(z))[0]

	int3 = integrate.quad(lambda z: int2(z), -l/2, l/2, epsabs=1.49e-3, epsrel=1.49e-3)[0]

	Bz_ave = int3/l**3
	return Bz_ave

# Plot polarization coil magnetic field strength
def StaticField_Plot(func_StaticField, title):

	Bz_ave = StaticField_ave(func_StaticField)

	print(title)
	print(np.round(Bz_ave*1000,decimals=4),"mT, ave")
	f_ave = Bz_ave/1000*gamma/2/np.pi # Larmor frequency
	print(np.round(f_ave,decimals=2), "Hz, Larmor frequency") 

	figure = plot.Figure()

	color = ["k", "b", "r"]
	Bz_min = 1e9
	Bz_max = 0
	i = 0
	for z in np.linspace(0, l/2, 3):

		Bz = func_StaticField(r,z)
		if abs(np.amax(Bz)) > Bz_max:
			Bz_max = abs(np.amax(Bz))
		if abs(np.amin(Bz)) < Bz_min:
			Bz_min = abs(np.amin(Bz))

		string = "$z = $" + str(np.round(z*1000,decimals=2)) + "$\\;\\rm{mm}$" 

		figure.plotFigure(x=r*1000, y=(Bz-Bz_ave)*1e6, linecolor=color[i], label=string)
		i = i+1
	
	print(np.round(Bz_max*1000,decimals=3), "mT, max", np.round(Bz_min*1000,decimals=3), "mT, min")
	ppm_max = (Bz_max - Bz_min)/(Bz_max)*1e6
	print(np.round(ppm_max,decimals=2), "ppm")
	print("\n")

	figure.setAxisLabels("$r \\quad (\\rm{mm})$", "$\\Delta B_{z} \\quad (\\mu\\rm{T})$", y_pad=4)
	figure.setAxis(0, 10, 2, np.floor((Bz_min-Bz_ave)*1e6), np.ceil((Bz_max-Bz_ave)*1e6), np.round((Bz_max-Bz_min)/4*1e6,decimals=0))
	figure.setLegend()
	string = "$B_{z,\\mathrm{ave}} = $" + str(np.round(Bz_ave*1000,decimals=2)) + "$\\; \\mathrm{mT}$"
	figure.annotateText(string, x_loc=0.95, y_loc=0.05, ha="right", va="bottom")
	figure.saveFigure(path + "/" + title + ".pdf")
	return Bz_ave, f_ave

Bz_ave, f_ave = StaticField_Plot(PolarizationCoil_Implemented, "PolarizationCoil")
StaticField_Plot(PolarizationCoil_Shim, "PolarizationCoil_Shim")
StaticField_Plot(PolarizationCoil_Michal, "PolarizationCoil_Michal")

# Cube for RF coil using transformation of r to x and y and 90 degree rotation
# of x and z about y axis because RF coil axis is 90 degrees rotated from polarizer
# coil axis.
def RFField_ave(func_RFField, I):

	def start(x,y,z):
		return func_RFField(np.sqrt(z**2+y**2),x, I) 

	def int1(y,z):
		return integrate.quad(start, -l/2, l/2, epsabs=1.49e-3, epsrel=1.49e-3,  args=(y,z))[0]

	def int2(z):
		return integrate.quad(int1, -l/2, l/2, epsabs=1.49e-3, epsrel=1.49e-3,  args=(z))[0]

	int3 = integrate.quad(lambda z: int2(z), -l/2, l/2,  epsabs=1.49e-3, epsrel=1.49e-3)[0]
	B1_ave = int3/l**3
	return B1_ave

# Plot RF coil magnetic field strength
def RFField_Plot(func_RFField, I_RF, title):

	B1_ave = RFField_ave(func_RFField, I_RF)

	print(title)
	print(np.round(B1_ave*1000,decimals=4),"mT, ave")
	print(np.round(B1_ave*1000/I_RF,decimals=4), "mT/A")

	figure = plot.Figure()

	color = ["k", "b", "r"]
	B1_min = 1e9
	B1_max = 0
	i = 0
	for z in np.linspace(0, l/2, 3):

		B1 = func_RFField(r,z,I_RF)
		if abs(np.amax(B1)) > B1_max:
			B1_max = abs(np.amax(B1))
		if abs(np.amin(B1)) < B1_min:
			B1_min = abs(np.amin(B1))

		string = "$z = $" + str(np.round(z*1000,decimals=2)) + "$\\;\\rm{mm}$" 

		figure.plotFigure(x=r*1000, y=(B1-B1_ave)*1e6, linecolor=color[i], label=string)
		i = i+1

	figure.setAxisLabels("$r \\quad (\\rm{mm})$", "$\\Delta B_1 \\quad (\\mu\\rm{T})$", y_pad=4)
	figure.setAxis(0, 10, 2, np.floor((B1_min-B1_ave)*1e6), np.ceil((B1_max-B1_ave)*1e6), np.round((B1_max-B1_min)*1e6)/2)
	figure.setLegend()
	string = "$B_{1,\\mathrm{ave}} = $" + str(np.round(B1_ave*1e6,decimals=2)) + "$\\; \\mu\\mathrm{T}$"
	figure.annotateText(string, x_loc=0.95, y_loc=0.05, ha="right", va="bottom")
	figure.saveFigure(path + "/" + title + ".pdf")

	print(np.round(B1_max*1000,decimals=3), "mT, max", np.round(B1_min*1000,decimals=3), "mT, min")
	ppm_max = (B1_max - B1_min)/(B1_max)*1e6
	print(np.round(ppm_max,decimals=2), "ppm")
	print("\n")

	return B1_ave

B1_ave = RFField_Plot(RFCoil_Solenoid, I_RF_Solenoid, "RFCoil_Solenoid")
RFField_Plot(RFCoil_Helmholtz, I_RF_Helmholtz, "RFCoil_Helmholtz")

l = 3e-2 # edge length of cube for sample in m
r = np.linspace(0, l/2, 1001) # radius to evaluate at in m

# Compare average of integral of products of static and RF field with products of average integrals of static and RF field to show valid approximation based on high uniformity
def FieldProduct(func_StaticField, func_RFField, I_RF, Bz_ave, B1_ave):

	def start(x,y,z):
		return func_RFField(np.sqrt(z**2+y**2),x,I_RF)*func_StaticField(np.sqrt(x**2+y**2),z)

	def int1(y,z):
		return integrate.quad(start, -l/2, l/2, epsabs=1.49e-3, epsrel=1.49e-3,  args=(y,z))[0]

	def int2(z):
		return integrate.quad(int1, -l/2, l/2, epsabs=1.49e-3, epsrel=1.49e-3,  args=(z))[0]

	int3 = integrate.quad(lambda z: int2(z), -l/2, l/2, epsabs=1.49e-3, epsrel=1.49e-3)[0]

	print('{:0.6e}'.format(int3/l**3), "average of integral of products")
	print('{:0.6e}'.format(B1_ave*Bz_ave), "products of average integrals")
	print(np.round(abs(int3-B1_ave*Bz_ave*l**3)/(int3)*100,decimals=4),"\b% error")
	print("\n")

FieldProduct(PolarizationCoil_Implemented, RFCoil_Solenoid, I_RF_Solenoid, Bz_ave, B1_ave)

def RFCurrent_Plot(f_transmit, I_RF, Q, transmit_width, shift_t, phase, min_f, max_f, data_path, title):

	TRANSMIT_SAMPLE_RATE = 10e6 # sample frequency in Hz
	TRANSMIT_WINDOW = 500e-3
	N_TRANSMIT = int(TRANSMIT_SAMPLE_RATE * TRANSMIT_WINDOW)

	t = np.linspace(-TRANSMIT_WINDOW/2, TRANSMIT_WINDOW/2, N_TRANSMIT)

	time_constant = 2*Q/(2*np.pi*f_transmit)

	growth_mask = (t > -transmit_width/2) & (t < -transmit_width/2 + time_constant*5)
	decay_mask = (t > transmit_width/2) & (t < transmit_width/2 + time_constant*5)
	pulse_mask = (t >= -transmit_width/2) & (t <= transmit_width/2 + time_constant*5)

	signal = np.empty_like(t)
	signal[:] = 0
	signal[pulse_mask] = I_RF*np.cos(2*np.pi*f_transmit*t+phase)[pulse_mask]
	signal[growth_mask] = signal[growth_mask]*(1-np.exp(-(t[growth_mask]+transmit_width/2)/time_constant))
	signal[decay_mask] = signal[decay_mask]*np.exp(-(t[decay_mask]-transmit_width/2)/time_constant)

	f = np.fft.fftfreq(N_TRANSMIT, 1/TRANSMIT_SAMPLE_RATE) # N points in array
	f_step = (np.max(f)-np.min(f))/N_TRANSMIT

	transmit_amplitude = 2/N_TRANSMIT*abs(np.fft.fft(signal))*TRANSMIT_WINDOW

	measured_transmit = np.loadtxt(path + '/' + data_path, delimiter=',', unpack=True, skiprows=2)[0:2]

	figure = plot.Figure()

	t = t + transmit_width/1.3
	figure.plotFigure(t[(t > 0) & (t < 2.3*transmit_width)]/1e-6, (signal*2)[(t > 0) & (t < 2.3*transmit_width)]/1e-3, linewidth=2, label='Simulated') # current re-doubled to account for both directions (which is true for measured signal)

	measured_mask = ((measured_transmit[0]-shift_t) > 0) & ((measured_transmit[0]-shift_t) < 2.2*transmit_width)
	measured_time = measured_transmit[0][measured_mask]
	measured_current = measured_transmit[1][measured_mask]

	figure.plotFigure((measured_time-shift_t)/1e-6, measured_current/1e-3, linewidth=0.5, linecolor='r', label='Measured')

	figure.setAxisLabels('$t \\quad (\\mu\\rm{s})$', '$I \\quad (\\rm{mA})$')
	y_step = np.round((np.max(signal*2/1e-3)-np.min(signal*2/1e-3))/3,decimals=0)
	x_max = np.round(np.max(transmit_width/1e-6*4/100))*100
	figure.setAxis(0, x_max, x_max/4, np.floor(np.min(signal*2/1e-3))-y_step/2, np.ceil(np.max(signal*2/1e-3))+y_step/2, y_step)

	figure.setInset()
	freq_mask = (f > min_f) & (f < max_f)
	figure.plotFigure(f[freq_mask]/1e3, (transmit_amplitude/np.max(transmit_amplitude))[freq_mask], axis="inset")
	figure.setAxis(min_f/1e3, max_f/1e3, (max_f/1e3-min_f/1e3)/2.5, 0, 1, 0.5, axis="inset")
	figure.setCustomAxis([min_f/1e3, f_transmit/1e3, max_f/1e3], [], axis="inset")
	figure.setAxisLabels('$f \\quad (\\mathrm{kHz})$', '', axis='inset')
	figure.setLegend(location='lower right', x_coord=0.95, y_coord=0)
	figure.annotateArrow(transmit_width/1e-6*1.75, y_loc=I_RF/1e-3/2, plot_num=1, x_scale=1.5, y_scale=1.3)
	figure.saveFigure(title + '.pdf')

	return f, f_step, transmit_amplitude

f, f_step, transmit_amplitude = RFCurrent_Plot(55e3, I_RF_Solenoid, Q_Solenoid, transmit_width_Solenoid, -84e-6, 0, 30e3, 80e3, 'TransmitPulse_Solenoid.csv', 'TransmitPulse_Solenoid')
RFCurrent_Plot(130e3, I_RF_Helmholtz, Q_Helmholtz, 108e-6, -299e-6, 3, 90e3, 170e3, 'TransmitPulse_Helmholtz.csv', 'TransmitPulse_Helmholtz')

print(gamma*transmit_amplitude[int(f_ave*1e3/f_step)]*B1_ave/I_RF_Solenoid*180/np.pi, "alpha angle (average)")
print(gamma*B1_ave*transmit_width_Solenoid*180/np.pi, "alpha angle (average simple)")
print(gamma*391e-6*15e-6*180/np.pi, "alpha angle (reference)")
print("\n")

def CalcFID(func_StaticField, func_RFField, I_RF, t):

	def argument(x,y,z):

		f_point = func_StaticField(np.sqrt(x**2+y**2),z)*gamma/(2*np.pi)
		alpha = gamma*transmit_amplitude[int(f_point/f_step)]*func_RFField(np.sqrt(z**2+y**2),x,I_RF)/I_RF

		B0 = func_StaticField(np.sqrt(x**2+y**2),z)
		M0 = gamma**2*h**2*B0*Ns/(16*np.pi**2*kb*T)

		scale = -1/I_RF*M0*np.sin(alpha)*func_RFField(np.sqrt(z**2+y**2),x,I_RF)

		time = np.multiply(np.exp(-t/T2)*(-1/T2),np.cos(2*np.pi*f_point*t))
		time += np.multiply(np.exp(-t/T2),-1*np.sin(2*np.pi*f_point*t)*2*np.pi*f_point)

		return scale*time

	def int1(y,z):
		return integrate.quad_vec(argument, -l/2, l/2, epsabs=1.49e-3, epsrel=1.49e-3, args=(y,z))[0]

	def int2(z):
		return integrate.quad_vec(int1, -l/2, l/2, epsabs=1.49e-3, epsrel=1.49e-3, args=(z,))[0]

	FID = integrate.quad_vec(int2, -l/2, l/2, epsabs=1.49e-3, epsrel=1.49e-3)[0]

	return FID

# %% FID Simulation in Frequency Domain
def SimulateFFT(func_StaticField, func_RFField, I_RF):

	echo_delay_time = 5e-3
	echo_sample_time = 4e-3
	sample_freq = 200e3

	NUM_AVERAGES = 40
	NUM_ECHOS = 250

	N_RECEIVE = int(sample_freq*echo_sample_time)
	t = np.linspace(0,echo_sample_time,N_RECEIVE)

	FID = CalcFID(func_StaticField, func_RFField, I_RF, t)

	V_max = max(FID)
	print(V_max/1e-6, "uV, max signal")
	print(V_max*1000*gain, "mV, max amplified signal")
	ADC_step = 330e-6
	print(V_max*gain/ADC_step, "ADC Steps")
	print("\n")

	noise_amplitude = 20e-3/gain # amount which matches measured -83 dBV average for 1 s worth of samples (no averaging) when trial simulated without any averaging

	START_TIME = 500e-6
	print(NUM_AVERAGES*(NUM_ECHOS*(echo_delay_time+500e-6)+20)/60, "min, total trial time")
	print("\n")
	N_CAPTURED = int(echo_sample_time*sample_freq*NUM_ECHOS)

	signal_FFT_ave = 0
	reference_FFT_ave = 0
	noise_FFT_ave = 0
	noise_reference_FFT_ave = 0

	f = np.fft.fftfreq(N_CAPTURED,1/sample_freq) # N points in array

	for i in list(range(NUM_AVERAGES)):

		for j in list(range(NUM_ECHOS)):

			noise = np.random.normal(0,noise_amplitude,N_RECEIVE)
			noise_reference = np.random.normal(0,noise_amplitude,N_RECEIVE)

			FID_delay_mask = (t > np.random.normal(0,echo_delay_time/10,1)[0])
			FID_delay = np.roll(np.where(FID_delay_mask,FID,0),-(np.size(FID_delay_mask)-np.sum(FID_delay_mask)))
			
			signal = FID_delay*np.exp(-j*echo_delay_time/T2)
			reference = FID_delay*np.exp(-(j+NUM_ECHOS)*echo_delay_time/T2)

			if (j == 0):
				signal_cat = signal
				reference_cat = reference
				noise_cat = noise
				noise_reference_cat = noise_reference

			else:
				signal_cat = np.concatenate([signal_cat,signal])
				reference_cat = np.concatenate([reference_cat,reference])
				noise_cat = np.concatenate([noise_cat,noise])
				noise_reference_cat = np.concatenate([noise_reference_cat,noise_reference])

		signal_FFT = 2/(N_CAPTURED)*abs(np.fft.fft(round_nearest(signal_cat*gain,ADC_step)))

		reference_FFT = 2/(N_CAPTURED)*abs(np.fft.fft(round_nearest(reference_cat*gain,ADC_step)))
		noise_FFT = 2/(N_CAPTURED)*abs(np.fft.fft(round_nearest(noise_cat*gain,ADC_step)))
		noise_reference_FFT = 2/(N_CAPTURED)*abs(np.fft.fft(round_nearest(noise_reference_cat*gain,ADC_step)))

		signal_FFT_ave = signal_FFT_ave + signal_FFT
		reference_FFT_ave = reference_FFT_ave + reference_FFT
		noise_FFT_ave = noise_FFT_ave + noise_FFT
		noise_reference_FFT_ave = noise_reference_FFT_ave + noise_reference_FFT

	signal_FFT_ave = signal_FFT_ave/NUM_AVERAGES
	reference_FFT_ave = reference_FFT_ave/NUM_AVERAGES
	noise_FFT_ave = noise_FFT_ave/NUM_AVERAGES
	noise_reference_FFT_ave = noise_reference_FFT_ave/NUM_AVERAGES

	signal_dBV_ave = 20*np.log10(signal_FFT_ave+noise_FFT_ave)
	reference_dBV_ave = 20*np.log10(reference_FFT_ave+noise_reference_FFT_ave)

	min_f = 47.5e3
	max_f = 57.5e3

	f_trimmed = f[(f > min_f) & (f < max_f)]
	signal_dBV_ave = signal_dBV_ave[(f > min_f) & (f < max_f)]
	reference_dBV_ave = reference_dBV_ave[(f > min_f) & (f < max_f)]

	signal_peak = np.max(signal_dBV_ave)
	noise_ave = np.sum(reference_dBV_ave)/np.size(f_trimmed)
	print(signal_peak, "dB, peak signal")
	print((f_trimmed[signal_dBV_ave == signal_peak])/1e3, "kHz, max frequency")
	print(noise_ave, "dB, ave noise")
	print(signal_peak-noise_ave, "dB, signal-noise difference")
	print("\n")

	figure = plot.Figure()

	figure.plotFigure(x=f_trimmed/1e3, y=signal_dBV_ave, linecolor='r', label='FID')
	figure.plotFigure(x=f_trimmed/1e3, y=reference_dBV_ave, label='Reference')
	figure.setAxisLabels('$f \\quad (\\rm{kHz})$', '$\\rm{Amplitude} \\quad (\\rm{dBV})$')
	figure.setAxis(min_f/1e3, max_f/1e3, 2.5, -87.5, -72.5, 5)
	figure.setLegend(custom_handles=[(1,),(0,)])
	figure.annotateArrow(x_loc=(f_trimmed[signal_dBV_ave == signal_peak])/1e3, plot_num=0, x_scale=2.5, y_scale=1, color='r', text='$V_{\\rm{peak}} = $' + "{:.2f}".format(signal_peak) + '$ \; \\rm{dBV}$')
	figure.annotateArrow(x_loc=(f_trimmed[signal_dBV_ave == signal_peak])/1e3, y_loc=noise_ave, plot_num=1, x_scale=2.5, y_scale=1.5, text='$V_{\\rm{ave}} = $' + "{:.2f}".format(noise_ave) + '$ \; \\rm{dBV}$')
	figure.saveFigure(path + "/ReceiveSignal_FFT.pdf")

SimulateFFT(PolarizationCoil_Implemented, RFCoil_Solenoid, I_RF_Solenoid)


def SimulateFID(func_StaticField, func_RFField, I_RF):

	echo_delay_time = 5e-3
	echo_sample_time = 4e-3
	sample_freq = 1e6

	NUM_AVERAGES = 40
	NUM_ECHOS = 250

	N_RECEIVE = int(sample_freq*echo_sample_time)
	t = np.linspace(0,echo_sample_time,N_RECEIVE)

	FID = CalcFID(func_StaticField, func_RFField, I_RF, t)

	figure = plot.Figure()

	figure.plotFigure(x=t/1e-3, y=FID/1e-6)
	figure.setAxis(0, 4, 1, -0.2, 0.4, 0.2)
	figure.setAxisLabels("$t \\quad (\\mathrm{ms})$", "$V \\quad (\\mu\\mathrm{V})$")
	figure.setInset()
	figure.plotFigure(x=t[t<100e-6]/1e-6, y=FID[t<100e-6]/1e-6, axis="inset")
	figure.setAxis(0, 100, 50, -0.2, 0.2, 0.1, axis="inset")
	figure.setAxisLabels("$t \\quad (\\mu\\mathrm{s})$", "$V \\quad (\\mu\\mathrm{V})$", axis="inset")
	figure.setCustomAxis([0, 50, 100], [-0.2, 0, 0.2], axis='inset')
	figure.annotateArrow(x_loc=0.1, y_loc=0.15, plot_num=1, x_scale=3, y_scale=1.5)
	figure.saveFigure(path + "/ReceiveSignal_Time.pdf")

SimulateFID(PolarizationCoil_Implemented, RFCoil_Solenoid, I_RF_Solenoid)
