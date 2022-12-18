import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import os

path = os.path.dirname(os.path.abspath(__file__))

plt.close('all')

# Need to replace commas with tabs in file
freqData = np.loadtxt(path + '/FreqData.txt', unpack=True, skiprows=1)[0:7]

Stage1 = np.loadtxt(path + '/Stage1.csv', delimiter=',', unpack=True, skiprows=1)[0:5]
Stage2 = np.loadtxt(path + '/Stage2.csv', delimiter=',', unpack=True, skiprows=1)[0:5]

f = freqData[0]
v1 = 20*np.log10(abs(freqData[1]+1j*freqData[2]))
v2 = 20*np.log10(abs(freqData[3]+1j*freqData[4]))

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

step_y = 25
step_x = 100

# ax.set_xlim(0-step_x/4, 500+step_x/4)
ax.set_ylim(-25-step_y/4, 100+step_y/4)

# ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(step_x))
# ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(step_x/2))

ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(step_y))
ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(step_y/2))

ax.set_xscale('log')

ax.set_xlabel('$f \\quad (\\rm{Hz})$', labelpad=0)
ax.set_ylabel('$\\rm{Gain} \\quad (\\rm{dB})$', labelpad=1)

mpl.rcParams['axes.unicode_minus'] = False # uses dash instead of minus for axis numbers


ax.plot(f[f <= 2e6], v1[f <= 2e6], linewidth=1, color='k', linestyle='dashed',markersize='2')
ax.plot(Stage1[1][Stage1[1] <= 2e6], Stage1[3][Stage1[1] <= 2e6], linewidth=1, color='k', linestyle='solid',markersize='2', label='Stage 1')
ax.plot(f[f <= 2e6], v2[f <= 2e6], linewidth=1, color='r', linestyle='dashed',markersize='2')
ax.plot(Stage2[1][Stage2[1] <= 2e6], Stage2[3][Stage2[1] <= 2e6], linewidth=1, color='r', linestyle='solid',markersize='2', label='Stage 2')

handles,labels = ax.get_legend_handles_labels()
# handles = [handles[1], handles[0]]
# labels = [labels[1], labels[0]]
ax.legend(handles,labels,frameon=False, bbox_to_anchor = [0.65, 0.02], loc='lower left', fontsize=8)


gain = Stage2[3][(Stage2[1] > 127e3) & (Stage2[1] < 130e3)]
print(gain)

ax.annotate(
	text="{:.2f}".format(float(gain)) + '$ \; \\rm{dB}$',
	fontsize=8,
	color='r',
	xy=(130e3,gain), 
	xytext=(30, 10), 
	xycoords='data', 
	textcoords='offset points', 
	arrowprops=dict(arrowstyle='<|-, head_width=0.15, head_length=0.3', lw=0.5, color='r', shrinkA=0, shrinkB=0, connectionstyle="angle3,angleA=0,angleB=90"),
	va='center', 
	ha='left',
)

plt.savefig(path + '/BodePlot.png', dpi=5000)

