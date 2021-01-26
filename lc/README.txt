Obs. ID     Transient  redshift  count_rate_net(pn,m1,m2)					duration(s)  counts(net, raw)	bin_size(s)		
0149780101: XT 030206    1.17	(7.15e-2, 4.725e-02, 5.099e-02)			1448		 1024					60	           
0203560201: XT 040610    0.5    (2.738e-03, 1.059e-03, 7.638e-04)    	26998		 288					800	
0300240501: XT 060207    0.3    (1.411e-01, 4.570e-02, 4.723e-02)    	238 		 137					12	
0300930301: XT 050925    0.3    (4.382e-03, 0, 2.102e-03)				4998		 82						160	
0502020101: XT 070618    0.37   (8.313e-01, 2.652e-01, 2.586e-01)    	163			 556					8	
0604740101: XT 100424    0.13   (6.406e-03, 0, 1.735e-03)				9998		 335					400	
0675010401: XT 110621    0.095  (3.026e-02, 9.509e-03, 8.332e-03)    	998			 385					48	
0743650701: XT 140811    0.57   (5.385e-03, 1.325e-03, 1.318e-03)    	9998		 163					400	
0760380201: XT 151128    0.48   (4.661e-03, 0., 1.190e-03)				5498		 130					160	
0765041301: XT 160220    0.3    (6.236e-03, 3.471e-03, 3.008e-03 )    	5998		 257					160	
0770380401: XT 151219    0.62   (5.916e-02, 0, 2.307e-02)				898			 264					24	
0781890401: XT 161028    0.29   (2.285e-01, 4.639e-02, 2.646e-02)    	171			 63						8	


fraction (net_count_rate/total; pn, m1, m2)
[(0.87, 0.925, 0.95), (0.71, 0.764, 0.756), (0.935, 0.89, 0.921), (0.752, 0, 0.8), (0.973, 0.936, 0.957), (0.689, 0., 0.685), 
(0.66, 0.669, 0.749), (0.729, 0.729, 0.816), (0.732, 0., 0.72), (0.699, 0.791, 0.811), (0.905, 0., 0.891), (0.94, 0.88, 0.911)]

N_H (Galactic, 10^20 cm^-2)
[6.4, 4.2, 10.8, 4.3, 1.6, 5.1, 3.0, 11.2, 5.1, 8.4, 2.5, 1.9]   

N_H (host, PL, 10^22 cm^-2)
[0.6, 0.4, 3.9, 2.4, 0.8, 0.7, 3.4, 3.7, 0.9, 0.5, 0.8, 2.2]

PL index
[3.4, 2.1, 1.9, 2.1, 3.2, 3.9, 2.9, 2.7, 1.9, 2.9, 3.5, 2.8]

BB temperature
[0.47, 0.68, 0.93, 0.45, 0.46, 0.13, 0.42, 0.32, 0.16, 0.45, 0.37, 0.41]

# results (redshift, counts)
criterion (s/n >= 5 and counts>50)
[(1.94, 72), (0.45, 254), (0.44, 50), (0.23, 109), (0.64, 49), (0.12, 201), (0.14, 78), (0.44, 160), (0.41, 136), (0.33, 139), (0.79, 49), (0.24, 50)]
[(1.91, 84), (0.47, 204), (0.47, 39), (0.2, 138), (0.69, 46), (0.15, 106), (0.17, 63), (0.46, 147), (0.42, 112), (0.3, 162), (0.72, 72), (0.24, 34)]

criterion (s/n >=5 and peak_flux > 2*mean_flux)
[(2.124623115577889, 65), (0.4854773869346734, 295), (0.4854773869346734, 37), (0.2477386934673367, 95), (0.7732663316582915, 33),(0.14763819095477387, 141), (0.16015075376884425, 52), (0.42291457286432166, 188), (0.3853768844221106, 106), (0.2977889447236181, 154), (0.7857788944723618, 53), (0.26025125628140705, 33)]

The lc folder contains the light curves with 3 columns: time (s), flux (counts s-1), flux (erg s-1 cm-2). The count rate is that from epiclccorr, so they are corrected for many instrumental effects and are effectively the count rate that would be observed by EPIC if the source was in the center of the FoV (see epiclccorr manual). I have combined pn+MOS1+MOS2 (although some of them lack MOS1 since it has a smaller FoV), let me know if you need them separately. You probably just need the light curve in physical units, which is not affected by combining the CCDs (normalized to cm-2).

The time zeropoint is probably not relevant for your simulations, but are relative to the Time in Table 1 of Alp & Larsson (2020). They are at a high temporal resolution of 4 s so that you can rebin to the cadence of your choice.

The bb and pl folders contain the best fit XSPEC models to the time-integrated spectra. bb are blackbody and pl are power law.
