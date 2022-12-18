import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import os

path = os.path.dirname(os.path.abspath(__file__))


plt.close('all')

timeData = np.loadtxt(path + '/TimeData.txt', unpack=True, skiprows=1)[0:3]

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

ax2 = ax.twinx()

ax.xaxis.set_tick_params(which='major', size=5, width=1, direction='in')
ax.xaxis.set_tick_params(which='minor', size=3, width=1, direction='in')

ax.yaxis.set_tick_params(which='major', size=5, width=1, direction='in')
ax.yaxis.set_tick_params(which='minor', size=3, width=1, direction='in')

ax2.yaxis.set_tick_params(which='major', size=5, width=1, direction='in')
ax2.yaxis.set_tick_params(which='minor', size=3, width=1, direction='in')

step_y1 = 0.2
step_y2 = 20
step_x = 10

ax.set_xlim(-2, 42)
t = timeData[0][timeData[0] < 40e-6]
v0 = timeData[1][timeData[0] < 40e-6]
v3 = timeData[2][timeData[0] < 40e-6]
ax.set_ylim(-0.4-step_y1/4, 0.4+step_y1/4)
ax2.set_ylim(-40-step_y2/4, 40+step_y2/4)

ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(step_x))
ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(step_x/2))

ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(step_y1))
ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(step_y1/2))

ax2.yaxis.set_major_locator(mpl.ticker.MultipleLocator(step_y2))
ax2.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(step_y2/2))

ax.set_xlabel('$t \\quad (\\mu\\rm{s})$', labelpad=1)
ax.set_ylabel('$V_{\\rm{in}} \\quad (\\mu\\rm{V})$', labelpad=4)
ax2.set_ylabel('$V_{\\rm{out}} \\quad (\\rm{mV})$', rotation=270, labelpad=9)

mpl.rcParams['axes.unicode_minus'] = False # uses dash instead of minus for axis numbers

ax.plot(t/1e-6, v0/1e-6, linewidth=1, color='k', linestyle='solid',markersize='2', label='Input')
ax2.plot(t/1e-6, v3/1e-3, linewidth=1, color='r', linestyle='solid', markersize='2', label='Output')

ax.annotate(
	text='',
	color='k',
	xy=(33.5,-0.3), 
	xytext=(23, -7), 
	xycoords='data', 
	textcoords='offset points', 
	arrowprops=dict(arrowstyle='<|-', lw=0.5, color='r', connectionstyle="angle3,angleA=0,angleB=90"),
	va='center', 
	ha='center'
)
ax.add_patch(Ellipse(
	xy=(33.5,-0.28), 
	width=4, 
	height=0.06,
	edgecolor='r', 
	fc='None', 
	lw=0.5
))

ax.annotate(
	text='',
	color='k',
	xy=(9,0.31), 
	xytext=(-23, 7), 
	xycoords='data', 
	textcoords='offset points', 
	arrowprops=dict(arrowstyle='<|-', lw=0.5, color='k', connectionstyle="angle3,angleA=0,angleB=90"),
	va='center', 
	ha='center'
)
ax.add_patch(Ellipse(
	xy=(9,0.29), 
	width=4, 
	height=0.06,
	edgecolor='k', 
	fc='None', 
	lw=0.5
))

plt.savefig(path + '/TimeData.pdf')



plt.close('all')

# Need to replace commas with tabs in file
freqData = np.loadtxt(path + '/FreqData.txt', unpack=True, skiprows=1)[0:7]

f = freqData[0]
v1 = 20*np.log10(abs(freqData[1]+1j*freqData[2]))
v2 = 20*np.log10(abs(freqData[3]+1j*freqData[4]))
v3 = 20*np.log10(abs(freqData[5]+1j*freqData[6]))

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

ax2.yaxis.set_tick_params(which='major', size=5, width=1, direction='in')
ax2.yaxis.set_tick_params(which='minor', size=3, width=1, direction='in')

step_y = 40

ax.set_ylim(-40-step_y/4, 120+step_y/4)

ax.set_xscale('log')


ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(step_y))
ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(step_y/2))


ax.set_xlabel('$f \\quad (\\rm{Hz})$', labelpad=1)
ax.set_ylabel('$\\rm{Gain} \\quad (\\rm{dB})$', labelpad=1)

mpl.rcParams['axes.unicode_minus'] = False # uses dash instead of minus for axis numbers

ax.plot(f[f <= 2e6], v1[f <= 2e6], linewidth=1, color='k', linestyle='solid',markersize='2', label='Stage 1')
ax.plot(f[f <= 2e6], v2[f <= 2e6], linewidth=1, color='b', linestyle='solid',markersize='2', label='Stage 2')
ax.plot(f[f <= 2e6], v3[f <= 2e6], linewidth=1, color='r', linestyle='solid',markersize='2', label='Stage 3')

handles,labels = ax.get_legend_handles_labels()
handles = [handles[2], handles[1], handles[0]]
labels = [labels[2], labels[1], labels[0]]
ax.legend(handles,labels,frameon=False, bbox_to_anchor = [0.65, 0.02], loc='lower left', fontsize=8)

plt.savefig(path + '/FreqData.pdf')