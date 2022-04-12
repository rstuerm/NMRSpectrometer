# %% Setup

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import scipy.special as sp
from scipy import integrate
from scipy.misc import derivative
import os

path = os.path.dirname(os.path.abspath(__file__))

# For quantization of values
def round_nearest(x, a):
    return np.round(x / a) * a

mu = 4*np.pi*1e-7 # permeability of free space in N/A^2
gamma = 267.522e6 # gyromagnetic ratio of hydrogen in rad/s/T
T2 = 2 # water spin relaxation time estimate in s
h = 6.62607004e-34 # Planck constant
kb = 1.38064852e-23 # Boltzmann constant
T = 305 # ~30 C in coil with shielding but no cooling and open top
Ns = 110.4*6.022e23/0.001 # proton density in water as protons/m^3 (110.4 M -> Mol/L, 1 L = 0.001 m^3)

I_RF = 72e-3/2/2 # Measured RF current amplitude, halved to account for only one polarization direction
gain = 50**2 # RF receive gain

l = 3e-2 # edge length of cube for sample in m
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


# Polarizer coil and RF Coil with fabrication parameters

def PolarizerCoil(r,z):

	R = 115e-3/2 # radius from outer diameter of 4" pipe for polarizer coil in m
	L = 300e-3 # length of polarizer coil in m
	gauge = 0.04*0.0254 # 18 AWG wire diameter in m
	R_shunt = 1.28 # shunt resistor for current measurement in ohms
	V_meas = 1.43 # voltage measured across shunt resistor in volts
	I_solenoid = V_meas/R_shunt
	shielding_loss = 1.07/1.14 
	Bz = solenoidB(r,z,R,L,1/gauge,I_solenoid)
	return Bz

def RFCoil(r,z):

	N = 56*6
	I = I_RF
	R = 2.75*0.0254/2
	L = 3*0.0254
	Bz = solenoidB(r,z,R,L,N/L,I)
	return Bz


# %% Static Field Calculation

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

color = ["k", "b", "r"]

Bz_min = 1e9
Bz_max = 0
i = 0
for z in np.linspace(0, l/2, 3):

	Bz = PolarizerCoil(r,z)
	if abs(np.amax(Bz)) > Bz_max:
		Bz_max = abs(np.amax(Bz))
	if abs(np.amin(Bz)) < Bz_min:
		Bz_min = abs(np.amin(Bz))

	string = "$z = $" + str(np.round(z*1000,decimals=2)) + "$\\;\\rm{mm}$" 
	ax.plot(r*1000,Bz*1000, linewidth=1, color=color[i], linestyle='solid', markersize='2', label=string)
	i = i+1

Bz_mid = (Bz_max + Bz_min)/2

step_y1 = 0.01
step_x = 2


mid_y1 = np.round(Bz_mid/1e-3 / step_y1 * 2) * step_y1 / 2

ax.set_xlim(-step_x/4, l/2/1e-2*10+step_x/4)

ax.set_ylim(mid_y1-3/2*step_y1-step_y1/4, mid_y1+step_y1+step_y1/4)

ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(step_x))
ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(step_x/2))

ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(step_y1))
ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(step_y1/2))


plt.xlabel("$r \\quad (\\rm{mm})$", labelpad=1)
plt.ylabel("$B_{z} \\quad (\\rm{mT})$", labelpad=4)

mpl.rcParams['axes.unicode_minus'] = False # uses dash instead of minus for axis numbers


handles,labels = ax.get_legend_handles_labels()
ax.legend(handles,labels,frameon=False, bbox_to_anchor = [0.55, 0.02], loc='lower left', fontsize=8)

plt.savefig(path + '/StaticField.pdf')

print("\n")
print("Static Field")
print(Bz_max*1000, "mT, max", Bz_min*1000, "mT, min")
ppm_max = (Bz_max - Bz_min)/(Bz_max)*1e6
print(ppm_max, "ppm")

# Cube for polarizer coil using transformation of r to x and y
def start(x,y,z):
	return PolarizerCoil(np.sqrt(x**2+y**2),z)

def int1(y,z):
	return integrate.quad(start, -l/2, l/2, epsabs=1.49e-3, epsrel=1.49e-3,  args=(y,z))[0]

def int2(z):
	return integrate.quad(int1, -l/2, l/2, epsabs=1.49e-3, epsrel=1.49e-3,  args=(z))[0]

int3 = integrate.quad(lambda z: int2(z), -l/2, l/2, epsabs=1.49e-3, epsrel=1.49e-3)[0]

Bz_ave = int3*1000/l**3
print(Bz_ave,"mT, ave")
f_ave = Bz_ave/1000*gamma/2/np.pi # Larmor frequency
print(f_ave, "Hz, Larmor frequency") 
print("\n")



# %% RF Field Calclation

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

color = ["k", "b", "r"]

B1_min = 1e9
B1_max = 0
i = 0
for z in np.linspace(0, l/2, 3):

	B1 = RFCoil(r,z)
	if abs(np.amax(B1)) > B1_max:
		B1_max = abs(np.amax(B1))
	if abs(np.amin(B1)) < B1_min:
		B1_min = abs(np.amin(B1))

	string = "$z = $" + str(np.round(z*1000,decimals=2)) + "$\\;\\rm{mm}$" 
	ax.plot(r*1000,B1/1e-6, linewidth=1, color=color[i], linestyle='solid', markersize='2', label=string)
	i = i+1

B1_mid = (B1_max + B1_min)/2

step_y1 = 2
step_x = 2


mid_y1 = np.round(B1_mid/1e-6 / step_y1 * 2) * step_y1 / 2

ax.set_xlim(-step_x/4, l/2/1e-2*10+step_x/4)

ax.set_ylim(mid_y1-4*step_y1-step_y1/4, mid_y1+2*step_y1+step_y1/4)

ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(step_x))
ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(step_x/2))

ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(step_y1))
ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(step_y1/2))


plt.xlabel("$r \\quad (\\rm{mm})$", labelpad=1)
plt.ylabel("$B_{1} \\quad (\\mu\\rm{T})$", labelpad=4)

mpl.rcParams['axes.unicode_minus'] = False # uses dash instead of minus for axis numbers

handles,labels = ax.get_legend_handles_labels()
ax.legend(handles,labels,frameon=False, bbox_to_anchor = [0.55, 0.02], loc='lower left', fontsize=8)

plt.savefig(path + '/RFField.pdf')

print("RF Field")
print(B1_max*1000, "mT, max", B1_min*1000, "mT, min")
ppm_max = (B1_max - B1_min)/(B1_max)*1e6
print(ppm_max, "ppm")


# Cube for RF coil using transformation of r to x and y and 90 degree rotation
# of x and z about y axis because RF coil axis is 90 degrees rotated from polarizer
# coil axis.
def start(x,y,z):
	return RFCoil(np.sqrt(z**2+y**2),x) 

def int1(y,z):
	return integrate.quad(start, -l/2, l/2, epsabs=1.49e-3, epsrel=1.49e-3,  args=(y,z))[0]

def int2(z):
	return integrate.quad(int1, -l/2, l/2, epsabs=1.49e-3, epsrel=1.49e-3,  args=(z))[0]

int3 = integrate.quad(lambda z: int2(z), -l/2, l/2,  epsabs=1.49e-3, epsrel=1.49e-3)[0]
RF_ave = int3*1000/l**3
print(RF_ave, "mT, ave")
print(RF_ave/I_RF, "mT/A")
print("\n")



# Product of Polarizer coil and rotated RF coil
def start(x,y,z):
	return RFCoil(np.sqrt(z**2+y**2),x)*PolarizerCoil(np.sqrt(x**2+y**2),z)

def int1(y,z):
	return integrate.quad(start, -l/2, l/2, epsabs=1.49e-3, epsrel=1.49e-3,  args=(y,z))[0]

def int2(z):
	return integrate.quad(int1, -l/2, l/2, epsabs=1.49e-3, epsrel=1.49e-3,  args=(z))[0]

intB = integrate.quad(lambda z: int2(z), -l/2, l/2, epsabs=1.49e-3, epsrel=1.49e-3)[0]

print(intB*1000*1000, "integral of products of B")
print(RF_ave*Bz_ave*l**3, "integral of products of average B")
print(round(abs(intB*1000*1000-RF_ave*Bz_ave*l**3)/(intB*1000*1000)*100,4),"\b% difference")
print("\n")


# %% Excitation Pulse Generation for Plotting

TRANSMIT_SAMPLE_RATE = 1e6 # sample frequency
f_transmit = 55e3 # excitation pulse frequency

TRANSMIT_WINDOW = 500e-3

N_TRANSMIT = int(TRANSMIT_SAMPLE_RATE * TRANSMIT_WINDOW)

t = np.linspace(-TRANSMIT_WINDOW/2, TRANSMIT_WINDOW/2, N_TRANSMIT)

transmit_width = 200e-6 # excitation pulse duration
Q = 7.9
time_constant = 2*Q/(2*np.pi*f_transmit)
print(time_constant, "s, time constant")
print("\n")

growth_mask = (t > -transmit_width/2) & (t < -transmit_width/2 + time_constant*5)
decay_mask = (t > transmit_width/2) & (t < transmit_width/2 + time_constant*5)
pulse_mask = (t >= -transmit_width/2) & (t <= transmit_width/2 + time_constant*5)

sig1 = np.empty_like(t)
sig1[pulse_mask] = I_RF*2*np.cos(2*np.pi*f_transmit*t)[pulse_mask] # doubled to account for both directions

sig1[growth_mask] = sig1[growth_mask]*(1-np.exp(-(t[growth_mask]+transmit_width/2)/time_constant))

sig1[decay_mask] = sig1[decay_mask]*np.exp(-(t[decay_mask]-transmit_width/2)/time_constant)


f = np.fft.fftfreq(N_TRANSMIT, 1/TRANSMIT_SAMPLE_RATE) # N points in array
transmit_amplitude = 2/N_TRANSMIT*abs(np.fft.fft(sig1))*TRANSMIT_WINDOW

measured_transmit = np.loadtxt(path + '/transmit.csv', delimiter=',', unpack=True, skiprows=2)[0:2]

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


step_y1 = 20
step_x = 250

ax.set_xlim(0-step_x/4, (5*transmit_width)/1e-6+step_x/4)

ax.set_ylim(-40-step_y1/4, 40+step_y1/4)

ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(step_x))
ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(step_x/2))

ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(step_y1))
ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(step_y1/2))


ax.set_xlabel('$t \\quad (\\mu\\rm{s})$', labelpad=1)
ax.set_ylabel('$I \\quad (\\rm{mA})$', labelpad=4)

mpl.rcParams['axes.unicode_minus'] = False # uses dash instead of minus for axis numbers

add_t = transmit_width
t = t + add_t
shift_t_meas = -71e-6
ax.plot(t[(t > 0) & (t < 3*transmit_width)]/1e-6, sig1[(t > 0) & (t < 3*transmit_width)]/1e-3, linewidth=2, color='k', linestyle='solid',markersize='2')
ax.plot((measured_transmit[0]+add_t+shift_t_meas)[(measured_transmit[0]+add_t+shift_t_meas > 0) & (measured_transmit[0]+add_t+shift_t_meas < 3*transmit_width)]/1e-6, measured_transmit[1][(measured_transmit[0]+add_t+shift_t_meas > 0) & (measured_transmit[0]+add_t+shift_t_meas < 3*transmit_width)]/1e-3, linewidth=0.5, color='r', linestyle='solid',markersize='2')

ax.plot(0,0, linewidth=1, color='k', linestyle='solid',markersize='2', label='Simulated')
ax.plot(0,0, linewidth=1, color='r', linestyle='solid',markersize='2', label='Measured')


inset = inset_axes(ax, width="30%", height=0.5, loc=1)

max_x = 80
min_x = 30

step_y1 = 0.5
step_x = 40
max_x = 80
min_x = 30

inset.set_xlim(min_x-step_x/4, max_x+step_x/4)

inset.set_ylim(0-step_y1/4, 1+step_y1/4)

inset.set_xticks([30, 55, 80])
inset.set_yticks([])

inset.set_xlabel('$f \\quad (\\rm{kHz})$', labelpad=1)

inset.plot(f[(f/1e3 > min_x) & (f/1e3 < max_x)]/1e3, (transmit_amplitude/max(transmit_amplitude))[(f/1e3 > min_x) & (f/1e3 < max_x)], linewidth=1, color='k', linestyle='solid',markersize='2', label='Input')

handles,labels = ax.get_legend_handles_labels()
ax.legend(handles,labels,frameon=False, bbox_to_anchor = [1, 0.02], loc='lower right', fontsize=8)

ax.annotate(
	text='',
	fontsize=8,
	color='r',
	xy=(450,5), 
	xytext=(22, 20), 
	xycoords='data', 
	textcoords='offset points', 
	arrowprops=dict(arrowstyle='<|-, head_width=0.15, head_length=0.3', lw=0.5, color='k', shrinkA=0, shrinkB=4, connectionstyle="angle3,angleA=0,angleB=90"),
	va='center', 
	ha='left',
)

plt.savefig(path + '/TransmitPulse.pdf')

# # %% Excitation Pulse Generation for Simulation

# TRANSMIT_SAMPLE_RATE = 1e6 # sample frequency
# f_transmit = 55e3 # excitation pulse frequency

# TRANSMIT_WINDOW = 500e-3

# N_TRANSMIT = int(TRANSMIT_SAMPLE_RATE * TRANSMIT_WINDOW)

# t = np.linspace(-TRANSMIT_WINDOW/2, TRANSMIT_WINDOW/2, N_TRANSMIT)

# transmit_width = 80e-6 # excitation pulse duration
# Q = 7.9
# time_constant = 2*Q/(2*np.pi*f_transmit)

# growth_mask = (t > -transmit_width/2) & (t < -transmit_width/2 + time_constant*5)
# decay_mask = (t > transmit_width/2) & (t < transmit_width/2 + time_constant*5)
# pulse_mask = (t >= -transmit_width/2) & (t <= transmit_width/2 + time_constant*5)

# sig1 = np.empty_like(t)
# sig1[pulse_mask] = I_RF*np.cos(2*np.pi*f_transmit*t)[pulse_mask]

# sig1[growth_mask] = sig1[growth_mask]*(1-np.exp(-(t[growth_mask]+transmit_width/2)/time_constant))

# sig1[decay_mask] = sig1[decay_mask]*np.exp(-(t[decay_mask]-transmit_width/2)/time_constant)


# f = np.fft.fftfreq(N_TRANSMIT, 1/TRANSMIT_SAMPLE_RATE) # N points in array
# transmit_amplitude = 2/N_TRANSMIT*abs(np.fft.fft(sig1))*TRANSMIT_WINDOW

f_step = (max(f)-min(f))/N_TRANSMIT

print(gamma*transmit_amplitude[int(f_ave/f_step)]*RF_ave*1e-3/I_RF*180/np.pi, "alpha angle (average)")
print(gamma*RF_ave*1e-3*transmit_width*180/np.pi, "alpha angle (average simple)")
print(gamma*391e-6*15e-6*180/np.pi, "alpha angle (reference)")
print("\n")

# %% FID Simulation in Frequency Domain

echo_delay_time = 5e-3
echo_sample_time = 4e-3
RECEIVE_SAMPLE_RATE = 200e3
N_RECEIVE = int(RECEIVE_SAMPLE_RATE*echo_sample_time)
t = np.linspace(0,echo_sample_time,N_RECEIVE)

def argument(x,y,z):
	f_point = PolarizerCoil(np.sqrt(x**2+y**2),z)*gamma/(2*np.pi)
	alpha = gamma*transmit_amplitude[int(f_point/f_step)]*RFCoil(np.sqrt(z**2+y**2),x)/I_RF

	B0 = PolarizerCoil(np.sqrt(x**2+y**2),z)
	M0 = gamma**2*h**2*B0*Ns/(16*np.pi**2*kb*T)

	scale = -1/I_RF*M0*np.sin(alpha)*RFCoil(np.sqrt(z**2+y**2),x)

	time = np.multiply(np.exp(-t/T2)*(-1/T2),np.cos(2*np.pi*f_point*t))
	time += np.multiply(np.exp(-t/T2),np.sin(2*np.pi*f_point*t)*2*np.pi*f_point)

	return scale*time

def int1(y,z):
	return integrate.quad_vec(argument, -l/2, l/2, epsabs=1.49e-3, epsrel=1.49e-3, args=(y,z))[0]

def int2(z):
	return integrate.quad_vec(int1, -l/2, l/2, epsabs=1.49e-3, epsrel=1.49e-3, args=(z,))[0]

FID = integrate.quad_vec(int2, -l/2, l/2, epsabs=1.49e-3, epsrel=1.49e-3)[0]

V_max = max(FID)
print(V_max/1e-6, "uV, max signal")
print(V_max*1000*gain, "mV, max amplified signal")
ADC_step = 330e-6
print(V_max*gain/ADC_step, "ADC Steps")
print("\n")

noise_amplitude = 20e-3/gain # based on measured -83 dBV average for 1 s worth of samples (no averaging)

f_res = 130e3 # sharp noise from resonance
resonance_amplitude = 1e-6 # based on measured -52 dBV test signal for 1 s worth of samples (no averaging)

NUM_AVERAGES = 40
NUM_ECHOS = 250

START_TIME = 500e-6
print(NUM_AVERAGES*(NUM_ECHOS*(echo_delay_time+500e-6)+20)/60, "min, total trial time")
print("\n")
N_CAPTURED = int(echo_sample_time*RECEIVE_SAMPLE_RATE*NUM_ECHOS)

signal_FFT_ave = 0
reference_FFT_ave = 0
noise_FFT_ave = 0
noise_reference_FFT_ave = 0
resonance_FFT_ave = 0

f = np.fft.fftfreq(N_CAPTURED,1/RECEIVE_SAMPLE_RATE) # N points in array

for i in list(range(NUM_AVERAGES)):

	for j in list(range(NUM_ECHOS)):

		resonance = resonance_amplitude*np.cos(2*np.pi*f_res*t + np.random.normal(0,2*np.pi,1)[0])
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
			resonance_cat = resonance

		else:
			signal_cat = np.concatenate([signal_cat,signal])
			reference_cat = np.concatenate([reference_cat,reference])
			noise_cat = np.concatenate([noise_cat,noise])
			noise_reference_cat = np.concatenate([noise_reference_cat,noise_reference])
			resonance_cat = np.concatenate([resonance_cat,resonance])

	signal_FFT = 2/(N_CAPTURED)*abs(np.fft.fft(round_nearest(signal_cat*gain,ADC_step)))
	reference_FFT = 2/(N_CAPTURED)*abs(np.fft.fft(round_nearest(reference_cat*gain,ADC_step)))
	noise_FFT = 2/(N_CAPTURED)*abs(np.fft.fft(round_nearest(noise_cat*gain,ADC_step)))
	noise_reference_FFT = 2/(N_CAPTURED)*abs(np.fft.fft(round_nearest(noise_reference_cat*gain,ADC_step)))
	resonance_FFT = 2/(N_CAPTURED)*abs(np.fft.fft(round_nearest(resonance_cat*gain,ADC_step)))

	signal_FFT_ave = signal_FFT_ave + signal_FFT
	reference_FFT_ave = reference_FFT_ave + reference_FFT
	noise_FFT_ave = noise_FFT_ave + noise_FFT
	noise_reference_FFT_ave = noise_reference_FFT_ave + noise_reference_FFT
	resonance_FFT_ave = resonance_FFT_ave + resonance_FFT

signal_FFT_ave = signal_FFT_ave/NUM_AVERAGES
reference_FFT_ave = reference_FFT_ave/NUM_AVERAGES
noise_FFT_ave = noise_FFT_ave/NUM_AVERAGES
resonance_FFT_ave = resonance_FFT_ave/NUM_AVERAGES
noise_reference_FFT_ave = noise_reference_FFT_ave/NUM_AVERAGES

signal_dBV_ave = 20*np.log10((signal_FFT_ave+noise_FFT_ave+resonance_FFT_ave)/1)
reference_dBV_ave = 20*np.log10((reference_FFT_ave+noise_reference_FFT_ave+resonance_FFT_ave)/1)

min_f = 50e3
max_f = 60e3

f_trimmed = f[(f > min_f) & (f < max_f)]
signal_dBV_ave = signal_dBV_ave[(f > min_f) & (f < max_f)]
reference_dBV_ave = reference_dBV_ave[(f > min_f) & (f < max_f)]

signal_peak = np.max(signal_dBV_ave)
noise_ave = np.sum(reference_dBV_ave)/np.size(f_trimmed)
print(signal_peak, "dB, peak signal")
print((f_trimmed[signal_dBV_ave >= signal_peak])/1e3, "kHz, max frequency")
print(noise_ave, "dB, ave noise")
print(signal_peak-noise_ave, "dB, signal-noise difference")
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


step_y1 = 10
step_x = 2.5


ax.set_xlim(min_f/1e3-step_x/4, max_f/1e3+step_x/4)

ax.set_ylim(-90-step_y1/4, -70+step_y1/4)

ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(step_x))
ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(step_x/2))

ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(step_y1))
ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(step_y1/2))


ax.set_xlabel('$f \\quad (\\rm{kHz})$', labelpad=1)
ax.set_ylabel('$\\rm{Amplitude} \\quad (\\rm{dBV})$', labelpad=4)

mpl.rcParams['axes.unicode_minus'] = False # uses dash instead of minus for axis numbers

ax.plot(f_trimmed/1e3, signal_dBV_ave, linewidth=1, color='r', linestyle='solid',markersize='2', label='FID')
ax.plot(f_trimmed/1e3, reference_dBV_ave, linewidth=1, color='k', linestyle='solid',markersize='2', label='Reference')

handles,labels = ax.get_legend_handles_labels()
handles = [handles[1], handles[0]]
labels = [labels[1], labels[0]]
ax.legend(handles,labels, frameon=False, bbox_to_anchor = [1, 0.02], loc='lower right', fontsize=8)

ax.annotate(
	text='$V_{\\rm{peak}} = $' + "{:.2f}".format(signal_peak) + '$ \; \\rm{dBV}$',
	fontsize=8,
	color='r',
	xy=(f_ave/1e3,signal_peak), 
	xytext=(20, 10), 
	xycoords='data', 
	textcoords='offset points', 
	arrowprops=dict(arrowstyle='<|-, head_width=0.15, head_length=0.3', lw=0.5, color='r', shrinkA=0, shrinkB=4, connectionstyle="angle3,angleA=0,angleB=90"),
	va='center', 
	ha='left',
)

ax.annotate(
	text='$V_{\\rm{ave}} = $' + "{:.2f}".format(noise_ave) + '$ \; \\rm{dBV}$',
	fontsize=8,
	color='k',
	xy=(f_ave/1e3,noise_ave), 
	xytext=(20, 23), 
	xycoords='data', 
	textcoords='offset points', 
	arrowprops=dict(arrowstyle='<|-, head_width=0.15, head_length=0.3', lw=0.5, color='k', shrinkA=0, shrinkB=4, connectionstyle="angle3,angleA=0,angleB=90"),
	va='center', 
	ha='left',
)

plt.savefig(path + '/ReceiveSignal_FFT.pdf')


# %% FID Simulation in Time Domain 


echo_delay_time = 5e-3
echo_sample_time = 4e-3
RECEIVE_SAMPLE_RATE = 1e6
N_RECEIVE = int(RECEIVE_SAMPLE_RATE*echo_sample_time)
t = np.linspace(0,echo_sample_time,N_RECEIVE)

def argument(x,y,z):
	f_point = PolarizerCoil(np.sqrt(x**2+y**2),z)*gamma/(2*np.pi)
	alpha = gamma*transmit_amplitude[int(f_point/f_step)]*RFCoil(np.sqrt(z**2+y**2),x)/I_RF

	B0 = PolarizerCoil(np.sqrt(x**2+y**2),z)
	M0 = gamma**2*h**2*B0*Ns/(16*np.pi**2*kb*T)

	scale = -1/I_RF*M0*np.sin(alpha)*RFCoil(np.sqrt(z**2+y**2),x)

	time = np.multiply(np.exp(-t/T2)*(-1/T2),np.cos(2*np.pi*f_point*t))
	time += np.multiply(np.exp(-t/T2),np.sin(2*np.pi*f_point*t)*2*np.pi*f_point)

	return scale*time

def int1(y,z):
	return integrate.quad_vec(argument, -l/2, l/2, epsabs=1.49e-3, epsrel=1.49e-3, args=(y,z))[0]

def int2(z):
	return integrate.quad_vec(int1, -l/2, l/2, epsabs=1.49e-3, epsrel=1.49e-3, args=(z,))[0]

FID = integrate.quad_vec(int2, -l/2, l/2, epsabs=1.49e-3, epsrel=1.49e-3)[0]

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


step_y1 = 0.1
max_y = 0.4
min_y = -0.2

step_x = 1
max_x = 4
min_x = 0

ax.set_xlim(min_x-step_x/4, max_x+step_x/4)

ax.set_ylim(min_y-step_y1/4, max_y+step_y1/4)

ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(step_x))
ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(step_x/2))

ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(step_y1))
ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(step_y1/2))


ax.set_xlabel('$t \\quad (\\rm{ms})$', labelpad=1)
ax.set_ylabel('$V \\quad (\\mu\\rm{V})$', labelpad=4)

mpl.rcParams['axes.unicode_minus'] = False # uses dash instead of minus for axis numbers


ax.plot(t/1e-3, FID/1e-6, linewidth=1, color='k', linestyle='solid',markersize='2', label='Signal')

inset = inset_axes(ax, width="30%", height=0.5, loc=1)

max_x = 100
min_x = 0

step_y1 = 0.1
step_x = 50

inset.set_xlim(min_x-step_x/4, max_x+step_x/4)
inset.set_ylim(-0.2-step_y1/4, 0.2+step_y1/4)

inset.set_xticks([0, 50, 100])
inset.set_yticks([-0.2, 0, 0.2])

inset.set_xlabel('$t \\quad (\\mu\\rm{s})$', labelpad=1)
ax.set_ylabel('$V \\quad (\\mu\\rm{V})$', labelpad=4)

inset.plot(t[(t/1e-6 > min_x) & (t/1e-6 < max_x)]/1e-6, FID[(t/1e-6 > min_x) & (t/1e-6 < max_x)]/1e-6, linewidth=1, color='k', linestyle='solid',markersize='2', label='Input')

ax.annotate(
	text='',
	fontsize=8,
	color='r',
	xy=(0.3,0.175), 
	xytext=(50, 20), 
	xycoords='data', 
	textcoords='offset points', 
	arrowprops=dict(arrowstyle='<|-, head_width=0.15, head_length=0.3', lw=0.5, color='k', shrinkA=0, shrinkB=4, connectionstyle="angle3,angleA=0,angleB=90"),
	va='center', 
	ha='left',
)

plt.savefig(path + '/ReceiveSignal.pdf')