import os 

instrument = ['pn', 'mos1', 'mos2']
filter_name = ['thin-5', 'thin', 'med-5', 'med', 'thick-5', 'thick']
aos = [7, 8, 10, 11, 13, 14, 16, 17, 18, 19, 20]
website_arf = "https://heasarc.gsfc.nasa.gov/FTP/webspec/arfs/"
website_rmf = "https://heasarc.gsfc.nasa.gov/FTP/webspec/rsps/"
website_pi = "https://heasarc.gsfc.nasa.gov/FTP/webspec/backgrounds/"
for ins in instrument:
	for filt in filter_name:
		for ao in aos:
			basename_arf = ins + '-' + filt + '-ao' + str(ao) + '.arf'
			basename_rmf = ins + '-' + filt + '-ao' + str(ao) + '.arf'
			basename_pi = ins + '-' + filt + '-ao' + str(ao) + '.pi'
			"""
			if not os.path.exists(basename_arf):
				command = "wget " + website_arf + basename_arf
				os.system(command)
			else:
				print(f"{basename_arf} exists!")
			"""
			if not os.path.exists(basename_rmf):
				command = "wget " + website_rmf + basename_rmf
				os.system(command)
			else:
				print(f"{basename_rmf} exists!")
			if not os.path.exists(basename_pi):
				command = "wget " + website_pi + basename_pi
				os.system(command)
			else:
				print(f"{basename_pi} exists!")

