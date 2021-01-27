"""This script is just used to store code that was used to test 'cal_sens', in case they would be useful in someday."""



def cal_flux(flux_s: float, flux_e: float, exposure: float, model: classmethod,
             pha=None, arf=None, rmf=None, bkg=None, mo_en_low=0.5, mo_en_hi=2.0,
             LiMa=False) -> float:
    """Find the detection flux limit for a given exposure and model

    Args:
        flux_s (float): starting flux
        flux_e (float): ending flux
        exposure (float): exposure time
        model (astropy model): the spectral model used to calculate flux

    Kwargs:
        pha (DATAPHA/None): the input source spectrum
        arf (DATAARF/None): effective area file of the instrument
        rmf (DATARMF/None): response matrix of the instrument
        bkg (DATAPHA/None): the background spectrum of the instrument
        mo_en_low (float): the lower energy boundary to calculate the sensitivity
        mo_en_hi (float): the upper energy boundary to calculate the sensitivity
        LiMa: use Lima formula  to estimate the ratio of signal to noise or not

    Returns:
        The flux limit that a source can be detected.

    References:
        Lima formula: Li & Ma, 1983 https://ui.adsabs.harvard.edu/abs/1983ApJ...272..317L/abstract
    """

    flux_del = np.log10(flux_e) - np.log10(flux_s)
    flux = 0.5 * (flux_s + flux_e)

    while flux_del > 0.001:
        # Set up the model and calculate the photon counts using the fakespec
        # function
        model.set_flux(flux, en_low=mo_en_low, en_hi=mo_en_hi)
        res = fakespec(pha=pha, arf=arf, rmf=rmf, bkg=bkg,
                       modelfunc=model.model, model_en_lo=mo_en_low,
                       model_en_hi=mo_en_hi, exposure=exposure)

        tot_cts = np.sum(res['counts'])
        bkg_cts = np.sum(res['bkg_counts'])
        src_cts = tot_cts - bkg_cts
        cts_5sig = poisson.ppf(0.999999, bkg_cts)

        # Two methods to calculate the detection
        # 1. Li-Ma formula
        # 2. Poisson statistic
        if LiMa:
            sn = (tot_cts - bkg_cts) / np.sqrt(tot_cts + bkg_cts)
            if sn >= 5.0:
                flux_e = 0.5 * (flux_s + flux_e)
            else:
                flux_s = 0.5 * (flux_s + flux_e)
        else:
            if src_cts <= 5.0:
                flux_s = 0.5 * (flux_s + flux_e)
            elif tot_cts > cts_5sig:
                flux_e = 0.5 * (flux_s + flux_e)
            else:
                flux_s = 0.5 * (flux_s + flux_e)

        flux = 0.5 * (flux_s + flux_e)
        flux_del = np.log10(flux_e) - np.log10(flux_s)

    return flux


def calc_snr(count_total, count_bkg, radius_src=40, radius_bkg=80, LiMa=False):
    """
    Notes:
        As for so-called LiMa equation, Details see Ti-Pei Li & Yu-Qian Ma 1983, ApJ, 272, 317
        http://articles.adsabs.harvard.edu/pdf/1983ApJ...272..317L
    Args:
        count_total: total count (rate)
        count_bkg: background count (rate)
        radius_src: the radius of source region, default 40"
        radius_bkg:  the radius of background region, default 80"
        LiMa: Using LiMa equation or not

    Returns:
        S/N: signal to noise ratio

    """
    alpha = np.square(radius_src) / np.square(radius_bkg)
    count_bkg /= alpha
    if LiMa:
        snr = np.sqrt(2) * np.sqrt(
            count_total * np.log((1 + alpha) / alpha * (count_total / (count_total + count_bkg)))
            + count_bkg * np.log((1 + alpha) * (count_bkg / (count_total + count_bkg))))
    else:
        count_bkg_scaled = count_bkg * alpha
        snr = count_total * 5 / poisson.ppf(0.999999, count_bkg_scaled)

    return snr


def get_bkg_count_rate(instrument: str, filter_name=None, threshold=15, radius=40):
    """

    Args:
        instrument: pn, mos1, or mos2.
        filter_name: thin, thick, or medium.
        threshold: see function 'get_random_lb', default is 15 degree.
        radius: the radius of source region, default is 40".

    Returns:
        mean, sigma of the background count rates
    """

    dir_bkg = './background/'
    if instrument == 'pn':
        path_bkg = dir_bkg + 'PN_bkg_info.txt'
    elif instrument == 'mos1':
        path_bkg = dir_bkg + 'MOS1_bkg_info.txt'
    elif instrument == 'mos2':
        path_bkg = dir_bkg + 'MOS2_bkg_info.txt'
    else:
        print('Error! The instrument should be pn, mos1, or mos2!')
        raise NameError

    bkg = Table.read(path_bkg, format='ascii')
    column_names = bkg.colnames
    bkg_dict = {}
    for i in column_names:
        bkg_dict[i] = np.array(bkg[i])
    df_bkg = pd.DataFrame(bkg_dict)
    df_bkg_full_window = df_bkg[df_bkg['SUBMODE'] != 'PrimeSmallWindow'].reset_index()

    # remove those with |b| < threshold, in unit of degree.
    lb = SkyCoord(df_bkg_full_window['ra_Deg']*u.degree,
                  df_bkg_full_window['dec_Deg']*u.degree,
                  frame='icrs').galactic
    l, b = lb.l.value, lb.b.value
    df_bkg_full_window_new = pd.concat([df_bkg_full_window,
                                        pd.DataFrame({"l_deg": l,
                                                      "b_deg": b})],
                                       axis=1, join='outer')
    df_bkg_full_window_new2 = df_bkg_full_window_new.drop(
        df_bkg_full_window_new[np.abs(df_bkg_full_window_new['b_deg']) <= threshold].index)

    # depend on filter_name
    if filter_name == 'medium':
        count_rate_bkg_raw = df_bkg_full_window_new2[df_bkg_full_window_new['FILTER'] == 'Medium']
    elif filter_name == 'thin':
        count_rate_bkg_raw = df_bkg_full_window_new2[df_bkg_full_window_new['FILTER'] == 'Thin1']
    else:
        count_rate_bkg_raw = df_bkg_full_window_new2

    radius_scaling_factor = np.square(count_rate_bkg_raw['radius_arcmin'] / radius)
    count_rate_bkg_mean = np.mean(count_rate_bkg_raw['count_rate'] / radius_scaling_factor)
    count_rate_bkg_sigma = np.std(count_rate_bkg_raw['count_rate'] / radius_scaling_factor)

    return count_rate_bkg_mean, count_rate_bkg_sigma


class MyThread(Thread):

    def __init__(self,func,args):
        Thread.__init__(self)
        self.func = func
        self.args = args
        self.res = []

    @property
    def getResult(self):
        return self.res

    def run(self):
        self.res.append(self.func(*self.args))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Define command line arguments
    parser.add_argument("--lc_file", type=str, default='./lc/0149780101_lc.txt', help="The path of light curve file")
    parser.add_argument("--rs_in", type=float, default=1.17, help="The redshift of this object")
    parser.add_argument("--satellite", type=str, default='xmm_newton', help="Name of the satellite. "
                                                                            "Default is xmm_newton")
    parser.add_argument("--instrument", type=str, default='pn', help="Name of the instrument, e.g., pn, "
                                                                     "mos1, or mos2. Default is pn.")
    parser.add_argument("--filter_name", type=str, default='thick-5', help="Filter in use. e.g., thin, "
                                                                           "med, or thick. "
                                                                           "Default is thin-5.")
    parser.add_argument("--ao", type=int, default=19, help="Name of the Announcement of Opportunity, "
                                                           "e.g., 17, 18, 19, 20. Default is 19.")
    parser.add_argument("--en_low", type=float, default=0.3, help="Lower limit for the energy range. "
                                                                  "Default is 0.3 keV.")
    parser.add_argument("--en_hi", type=float, default=10.0, help="Upper limit for the energy range. "
                                                                  "Default is 10.0 keV")
    parser.add_argument("--mo_en_low", type=float, default=0.3, help="Lower limit for the energy range. "
                                                                     "Default is 0.3 keV.")
    parser.add_argument("--mo_en_hi", type=float, default=10.0, help="Upper limit for the energy range. "
                                                                     "Default is 10.0 keV.")
    parser.add_argument("--mo_lumi", type=float, default=1.E45, help="Luminosity over the model energy "
                                                                     "range. Default is 1e45 erg/s.")

    args = parser.parse_args()

    # lc_file = args.lc_file
    # rs_in = args.rs_in
    satellite = args.satellite
    instrument = args.instrument
    filter_name = args.filter_name
    ao = args.ao
    en_hi = args.en_hi
    en_low = args.en_low
    mo_en_hi = args.mo_en_hi
    mo_en_low = args.mo_en_low
    mo_lumi = args.mo_lumi

    paths = glob.glob('./lc/*.txt')
    paths.sort()
    time_start = time.time()

    df_infos = get_obs_infos()
    obs_ids = np.array(df_infos['obs_id'], dtype=str)
    obj_names = np.array(df_infos['obj_name'])
    bin_sizes = np.array(df_infos['bin_size'])
    rs_list = np.array(df_infos['redshift'])

    nh_gal_list = np.array(df_infos['nh_gal']) * 0.01
    nh_host_list = np.array(df_infos['nh_host'])
    alpha_list = np.array(df_infos['pl_index'])
    temperature_list = np.array(df_infos['bb_T']) * 1e3

    """
    # This part is used to check the program
    mos1 = (True, True, True, False, True, False, True, True, False, True, False, True)
    generate_light_curve()
    generate_light_curve(instrument='mos1')
    generate_light_curve(instrument='mos2')
    for obs_id, obj_name, m1, bin_size in zip(obs_ids, obj_names, mos1, bin_sizes):
        test = plot_lc_total(obs_id, title=obj_name, mos1=m1, bin_size=bin_size)
    """

    res_dir = './results/'
    rs_out = []
    instruments_list = [('pn', 'mos1', 'mos2'),
                        ('pn', 'mos1', 'mos2'),
                        ('pn', 'mos1', 'mos2'),
                        ('pn', 'mos2'),
                        ('pn', 'mos1', 'mos2'),
                        ('pn', 'mos2'),
                        ('pn', 'mos1', 'mos2'),
                        ('pn', 'mos1', 'mos2'),
                        ('pn', 'mos2'),
                        ('pn', 'mos1', 'mos2'),
                        ('pn', 'mos2'),
                        ('pn', 'mos1', 'mos2')]

    instruments_list2 = [['pn'],
                         ['pn'],
                         ['pn'],
                         ['pn'],
                         ['pn'],
                         ['pn'],
                         ['pn'],
                         ['pn'],
                         ['pn'],
                         ['pn'],
                         ['pn'],
                         ['pn']]
    instruments_list3 = [('pn', 'mos1', 'mos2'),
                        ('pn', 'mos1', 'mos2'),
                        ('pn', 'mos1', 'mos2'),
                        ('pn', 'mos1', 'mos2'),
                        ('pn', 'mos1', 'mos2'),
                        ('pn', 'mos1', 'mos2'),
                        ('pn', 'mos1', 'mos2'),
                        ('pn', 'mos1', 'mos2'),
                        ('pn', 'mos1', 'mos2'),
                        ('pn', 'mos1', 'mos2'),
                        ('pn', 'mos1', 'mos2'),
                        ('pn', 'mos1', 'mos2')]
    for i, lc_file, rs_in, bin_size, instruments in zip(range(len(paths)), paths, rs_list, bin_sizes, instruments_list3):

        # load the calibration matrix
        print("light curve file:" + lc_file)
        print(f"initial redshift: {rs_in}")
        # pha, arf, rmf, bkg = GetResp(satellite=satellite, instrument=instrument,
        #                              filter_name=filter_name, ao=ao).response

        rs = cal_redshift(lc_file, rs_in, satellite=satellite, instrument=instruments,
                          filter_name=filter_name, ao=ao, bin_size=bin_size,
                          save_res=True, plot_res=True, LiMa=True, poisson=True)

        #rs = cal_redshift(lc_file, rs_in, satellite=satellite, instrument=instruments, filter_name=filter_name,
        #                  ao=ao, nh_gal=nh_gal_list[i], nh_host=nh_host_list[i], alpha=alpha_list[i],
        #                  temperature=temperature_list[i], bin_size=bin_size, save_res=False, plot_res=False,
        #                  LiMa=True, index_obj=i)

        rs_out.append(rs)

    print('finished!')
    print(rs_out)
    time_end = time.time()
    time_delta = time_end - time_start
    print(f'Times Used: {time_delta} s')