Obs. ID     Transient
0149780101: XT 030206
0203560201: XT 040610
0300240501: XT 060207
0300930301: XT 050925
0502020101: XT 070618
0604740101: XT 100424
0675010401: XT 110621
0743650701: XT 140811
0760380201: XT 151128
0765041301: XT 160220
0770380401: XT 151219
0781890401: XT 161028

The lc folder contains the light curves with 3 columns: time (s), flux (counts s-1), flux (erg s-1 cm-2). The count rate is that from epiclccorr, so they are corrected for many instrumental effects and are effectively the count rate that would be observed by EPIC if the source was in the center of the FoV (see epiclccorr manual). I have combined pn+MOS1+MOS2 (although some of them lack MOS1 since it has a smaller FoV), let me know if you need them separately. You probably just need the light curve in physical units, which is not affected by combining the CCDs (normalized to cm-2).

The time zeropoint is probably not relevant for your simulations, but are relative to the Time in Table 1 of Alp & Larsson (2020). They are at a high temporal resolution of 4 s so that you can rebin to the cadence of your choice.

The bb and pl folders contain the best fit XSPEC models to the time-integrated spectra. bb are blackbody and pl are power law.
