import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


plt.figure()
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'figure.dpi': 300})
plt.rcParams.update({'savefig.dpi': 300})
plt.rcParams.update({'mathtext.fontset': 'cm'})

pdf = PdfPages('images/scaling.pdf')
fontLegend = 16
fontAxis = 16
fontText = 16

# Dopri5:
latD = np.array([ \
2.904486077, 1.923772811, 0.963700428,  # 1e-1
3.503132770, 2.325528942, 1.142332049,  # 1e-2
4.399596091, 2.760949737, 1.351919522,  # 1e-3
7.323495498, 4.878686175, 2.385255797]) # 1e-4

fidD = np.array([ \
6.084561348, 7.005294800, 13.79725742,
5.883783817, 7.067956448, 14.45279121,
5.751778126, 7.091618538, 15.14557552,
5.896519184, 7.308739662, 15.69930935])

latD = np.reshape(latD, (4,3)).T
fidD = np.reshape(fidD, (4,3)).T
latD = latD[:,1] #[2:,:-2]
fidD = fidD[:,1] #[2:,:-2]

# Euler:
# 12, 08, 04, SM
latE = np.array([ \
0.131764306, 0.096437792, 0.064267033, 0.127288711,  # 1
0.204945960, 0.135275196, 0.067246705, 0.200186887,  # 2
0.410422194, 0.271347597, 0.134881787, 0.401605042,  # 4
0.816952790, 0.543261729, 0.269963419, 0.803603834,  # 8
1.635907493, 1.087727956, 0.540702272, 1.608782623,  # 16
3.272716141, 2.175870214, 1.081549636, 3.218326182,  # 32
6.552376361, 4.349655228, 2.164436358, 6.435472863,  # 64
13.13659565, 8.701929774, 4.329354476, 12.88829373]) # 128

fidE = np.array([ \
25.28643036, 24.30469322, 30.96321869, 24.13863754,
22.87129211, 21.99527550, 22.60784912, 21.70017624,
20.09344482, 19.37377930, 19.44503021, 19.18571472,
16.29567337, 15.76149368, 16.38772964, 15.91781712,
12.68265533, 12.45587063, 13.96543980, 12.52924156,
9.573678970, 9.839547157, 12.58811855, 9.514616013,
7.126485825, 8.130121231, 12.57286453, 7.072010040,
5.896785736, 7.868789673, 14.21733093, 6.015111446])

latE = np.reshape(latE, (8,4)).T
fidE = np.reshape(fidE, (8,4)).T
latE = latE[:,:-2]
fidE = fidE[:,:-2]

k = ['-gx', '-bo', '-mh', '--y.', '-cD']
#plt.plot(latE[0], fidE[0], k[0], markersize=6, linewidth=1.5, label=r'$\mathrm{ODE}_{T \mid l,d} (\mathrm{ODE}_{l=12} )$')  #, ':r', label=r[3], markersize=6, linewidth=1.5)
plt.plot(latE[3], fidE[3], k[3], markersize=6, linewidth=1.5, label=r'SM $(\mathrm{ODE}_{T \mid d})$')  #, ':r', label=r[3], markersize=6, linewidth=1.5)
plt.plot(latE[1], fidE[1], k[1], markersize=6, linewidth=1.5, label=r'$\mathrm{ODE}_{T \mid l,d} (\mathrm{ODE}_{l=8} )$')  #, ':r', label=r[3], markersize=6, linewidth=1.5)
plt.plot(latE[2], fidE[2], k[2], markersize=6, linewidth=1.5, label=r'$\mathrm{ODE}_{T \mid l,d} (\mathrm{ODE}_{l=4} )$')  #, ':r', label=r[3], markersize=6, linewidth=1.5)
plt.plot(latD,    fidD,   k[-1], markersize=6, linewidth=1.5, label=r'$\mathrm{ODE}_{t \mid l,d} (\mathrm{ODE}_{l=8} )$') #, ':r', label=r[3], markersize=6, linewidth=1.5)

plt.xlim(0, 3)
plt.ylim(0, 32)
plt.grid(True)
plt.xlabel('Latency, sec')
plt.ylabel('FID-50K')
plt.legend(loc='upper right', shadow=False, fontsize = fontLegend, ncol=1)
plt.show()
plt.savefig('images/scaling.png', bbox_inches='tight')

plt.savefig('images/scaling.svg', format="svg", bbox_inches='tight')
pdf.savefig(bbox_inches='tight')
pdf.close()
print('Plots done!')


l4t1   = np.array([32.38871257,	32.0515213,	32.60470581])
l4t4   = np.array([22.40657679,	22.23408127,	22.49620438])
l4t128 = np.array([15.40200011,	15.29460812,	15.48247623])

l8t1   = np.array([27.71744537,	27.29597664,	27.94865036])
l8t4   = np.array([23.29220263,	22.96026993,	23.68478775])
l8t128 = np.array([8.833935738,	8.788946152,	8.869131088])

l12t1   = np.array([29.69008954,	29.26315689,	29.97071838])
l12t4   = np.array([24.11519432,	23.58354568,	24.42061615])
l12t128 = np.array([6.874435584,	6.790030479,	6.946063519])

print('{:.2f} / {:.2f} / {:.2f}'.format(np.std(l4t128), np.std(l4t4), np.std(l4t1)))
print('{:.2f} / {:.2f} / {:.2f}'.format(np.std(l8t128), np.std(l8t4), np.std(l8t1)))
print('{:.2f} / {:.2f} / {:.2f}'.format(np.std(l12t128), np.std(l12t4), np.std(l12t1)))
