"""
Data
----

Define some datasets as functions. In each case, an instance of :class:`Sample`
is returned to have a consistant representation.

* :func:`el_nino`,
* :func:`tahiti`,
* :func:`mascaret`,
* :func:`marthe`.
"""
import os
import numpy as np
from batman.space import Sample


# Common path
PATH = os.path.dirname(os.path.realpath(__file__))


def el_nino():
    """El Nino dataset."""
    desc = ("Averaged monthly sea surface temperature (SST) in degrees Celcius"
            " of the Pacific Ocean at 0-10 deg South and 90-80 deg West"
            " between 1950 and 2007.\nSource: NOAA - ERSSTv5 - Nino 1+2 at"
            " http://www.cpc.ncep.noaa.gov/data/indices/")

    labels, data = np.loadtxt(os.path.join(PATH, 'elnino.dat'),
                              skiprows=1, usecols=(0, 2), unpack=True)
    labels = labels.reshape(-1, 12)[:, 0].reshape(-1, 1)
    data = data.reshape(-1, 12)

    flabels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    sample = Sample(space=labels, data=data, plabels=['Year'], flabels=flabels)
    sample.desc = desc
    return sample


def tahiti():
    """Tahiti dataset."""
    desc = ("Averaged monthly sea level pressure (SLP) in millibars"
            "at Tahiti between 1951 and 2016.\nSource: NOAA - Tahiti SLP at"
            " http://www.cpc.ncep.noaa.gov/data/indices/")

    dataset = np.loadtxt(os.path.join(PATH, 'tahiti.dat'),
                         skiprows=4, usecols=range(0, 13))

    labels = dataset[:, 0].reshape(-1, 1)
    data = dataset[:, 1:]

    flabels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    sample = Sample(space=labels, data=data, plabels=['Year'], flabels=flabels)
    sample.desc = desc
    return sample


def mascaret(multizone=False, fname=None):
    """Mascaret dataset.

    :param bool multizone: Use only one global Ks or 3 Ks.
    :param list(str) fname: Path to the file that contains input and
      output data. File format must be `npy`.
    """
    desc = ("Monte-Carlo sampling simulated using 1D MASCARET flow solver."
            " The Garonne river was used and the output consists in water"
            " height observations. There are two dataset: (i) 5000 samples,"
            " multizone with 3 different friction coefficients"
            " Ks(1,2,3)~U(15, 60) discretized on 463 points along the channel;"
            " (ii) 100000 samples, single zone with Ks~U(15, 60)  discretized"
            " in 14. In both cases the mass flow rate Q~N(4035, 400).")

    if multizone:
        flabels = ['13150.0', '13250.0', '13350.0', '13450.0', '13550.0',
                   '13650.0', '13750.0', '13850.0', '13950.0', '14025.0',
                   '14128.333333333334', '14231.666666666668', '14335.0',
                   '14448.333333333334', '14561.666666666668', '14675.0',
                   '14780.0', '14885.0', '14990.0', '15095.0', '15200.0',
                   '15312.5', '15425.0', '15537.5', '15650.0', '15762.5',
                   '15875.0', '15981.25', '16087.5', '16193.75', '16300.0',
                   '16406.25', '16512.5', '16618.75', '16725.0', '16830.833333333332',
                   '16936.666666666664', '17042.499999999996', '17148.33333333333',
                   '17254.16666666666', '17360.0', '17500.0', '17640.0', '17750.0',
                   '17860.0', '17970.0', '18080.0', '18190.0', '18300.0',
                   '18403.571428571428', '18507.142857142855', '18610.714285714283',
                   '18714.28571428571', '18817.857142857138', '18921.428571428565',
                   '19025.0', '19131.25', '19237.5', '19343.75', '19450.0', '19556.25',
                   '19662.5', '19768.75', '19875.0', '19979.166666666668', '20083.333333333336',
                   '20187.500000000004', '20291.66666666667', '20395.83333333334',
                   '20500.0', '20603.125', '20706.25', '20809.375', '20912.5',
                   '21015.625', '21118.75', '21221.875', '21325.0', '21425.0',
                   '21525.0', '21625.0', '21725.0', '21825.0', '21925.0', '22032.0',
                   '22139.0', '22246.0', '22353.0', '22460.0', '22576.25', '22692.5',
                   '22808.75', '22925.0', '23031.5', '23138.0', '23244.5', '23351.0',
                   '23457.5', '23564.0', '23670.5', '23777.0', '23883.5', '23990.0',
                   '24110.0', '24230.0', '24350.0', '24455.0', '24560.0', '24665.0',
                   '24770.0', '24875.0', '24975.0', '25075.0', '25175.0', '25275.0',
                   '25375.0', '25475.0', '25575.0', '25675.0', '25775.0', '25875.0',
                   '25975.0', '26075.0', '26175.0', '26275.0', '26383.333333333332',
                   '26491.666666666664', '26599.999999999996', '26708.33333333333',
                   '26816.66666666666', '26924.999999999993', '27033.333333333325',
                   '27141.666666666657', '27250.0', '27359.375', '27468.75', '27578.125',
                   '27687.5', '27796.875', '27906.25',
                   '28015.625', '28125.0', '28240.0', '28355.0', '28470.0', '28585.0',
                   '28700.0', '28810.0', '28920.0', '29030.0', '29140.0', '29250.0',
                   '29360.0', '29463.0', '29566.0', '29669.0', '29772.0', '29875.0',
                   '29978.0', '30081.0', '30184.0', '30287.0', '30390.0', '30491.0',
                   '30592.0', '30693.0', '30794.0', '30895.0', '30996.0', '31097.0',
                   '31198.0', '31299.0', '31400.0', '31505.0', '31610.0', '31715.0',
                   '31820.0', '31830.0', '31990.0', '32000.0', '32075.0', '32177.14285714286',
                   '32279.285714285717', '32381.428571428576',
                   '32483.571428571435', '32585.714285714294', '32687.857142857152',
                   '32790.0', '32904.166666666664', '33018.33333333333', '33132.49999999999',
                   '33246.66666666666', '33360.83333333332', '33475.0', '33582.142857142855',
                   '33689.28571428571', '33796.428571428565', '33903.57142857142',
                   '34010.714285714275', '34117.85714285713', '34225.0',
                   '34332.142857142855', '34439.28571428571', '34546.428571428565',
                   '34653.57142857142', '34760.714285714275', '34867.85714285713',
                   '34975.0', '35077.5', '35180.0', '35282.5', '35385.0', '35487.5',
                   '35590.0', '35698.333333333336', '35806.66666666667', '35915.00000000001',
                   '36023.33333333334', '36131.66666666668', '36240.0', '36290.0', '36340.0',
                   '36441.666666666664', '36543.33333333333', '36644.99999999999',
                   '36746.66666666666', '36848.33333333332', '36950.0', '37066.666666666664',
                   '37183.33333333333', '37300.0', '37408.333333333336', '37516.66666666667',
                   '37625.0', '37725.0', '37825.0', '37926.36363636364', '38027.72727272728',
                   '38129.09090909092', '38230.45454545456', '38331.8181818182',
                   '38433.18181818184', '38534.54545454548', '38635.90909090912',
                   '38737.27272727276', '38838.6363636364', '38940.0', '39041.666666666664',
                   '39143.33333333333', '39244.99999999999', '39346.66666666666',
                   '39448.33333333332', '39550.0', '39650.0', '39750.0', '39850.0',
                   '39950.0', '40051.666666666664', '40153.33333333333', '40254.99999999999',
                   '40356.66666666666', '40458.33333333332', '40560.0', '40663.0', '40766.0',
                   '40869.0', '40972.0', '41075.0', '41178.0', '41281.0', '41384.0', '41487.0',
                   '41590.0', '41700.0', '41810.0', '41920.0', '42030.0', '42140.0', '42247.0',
                   '42354.0', '42461.0', '42568.0', '42675.0', '42793.75', '42912.5', '43031.25',
                   '43150.0', '43262.5', '43375.0', '43487.5', '43600.0', '43712.5', '43825.0',
                   '43929.166666666664', '44033.33333333333', '44137.49999999999',
                   '44241.66666666666', '44345.83333333332', '44450.0', '44557.5',
                   '44665.0', '44772.5', '44880.0', '44987.5', '45095.0', '45202.5',
                   '45310.0', '45418.333333333336', '45526.66666666667', '45635.00000000001',
                   '45743.33333333334', '45851.66666666668', '45960.0', '46076.0',
                   '46192.0', '46308.0', '46424.0', '46540.0', '46650.625', '46761.25',
                   '46871.875', '46982.5', '47093.125', '47203.75', '47314.375',
                   '47425.0', '47533.125', '47641.25', '47749.375', '47857.5',
                   '47965.625', '48073.75', '48181.875', '48290.0', '48393.333333333336',
                   '48496.66666666667', '48600.00000000001', '48703.33333333334',
                   '48806.66666666668', '48910.0', '49015.555555555555', '49121.11111111111',
                   '49226.666666666664', '49332.22222222222', '49437.777777777774',
                   '49543.33333333333', '49648.88888888888', '49754.44444444444',
                   '49860.0', '49965.0', '50070.0', '50175.0', '50280.0', '50385.0',
                   '50490.0', '50601.666666666664', '50713.33333333333', '50825.0',
                   '50939.166666666664', '51053.33333333333', '51167.49999999999',
                   '51281.66666666666', '51395.83333333332', '51510.0', '51620.833333333336',
                   '51731.66666666667', '51842.50000000001', '51953.33333333334',
                   '52064.16666666668', '52175.0', '52291.25', '52407.5', '52523.75',
                   '52640.0', '52744.375', '52848.75', '52953.125', '53057.5',
                   '53161.875', '53266.25', '53370.625', '53475.0', '53591.666666666664',
                   '53708.33333333333', '53825.0', '53967.5', '54110.0', '54211.875',
                   '54313.75', '54415.625', '54517.5', '54619.375', '54721.25',
                   '54823.125', '54925.0', '55034.375', '55143.75', '55253.125',
                   '55362.5', '55471.875', '55581.25', '55690.625', '55800.0',
                   '55905.0', '56010.0', '56115.0', '56220.0', '56325.0', '56428.125',
                   '56531.25', '56634.375', '56737.5', '56840.625', '56943.75',
                   '57046.875', '57150.0', '57250.0', '57350.0', '57450.0', '57550.0',
                   '57650.0', '57750.0', '57850.0', '57957.142857142855',
                   '58064.28571428571', '58171.428571428565', '58278.57142857142',
                   '58385.714285714275', '58492.85714285713', '58600.0', '58712.0',
                   '58824.0', '58936.0', '59048.0', '59160.0', '59266.92307692308',
                   '59373.846153846156', '59480.769230769234', '59587.69230769231',
                   '59694.61538461539', '59801.53846153847', '59908.461538461546',
                   '60015.384615384624', '60122.3076923077', '60229.23076923078',
                   '60336.15384615386', '60443.07692307694', '60550.0', '60654.545454545456',
                   '60759.09090909091', '60863.63636363637', '60968.18181818182',
                   '61072.72727272728', '61177.272727272735', '61281.81818181819',
                   '61386.36363636365', '61490.9090909091', '61595.45454545456',
                   '61700.0', '61818.75', '61937.5', '62056.25', '62175.0']

        plabels = ['Ks1', 'Ks2', 'Ks3', 'Q']
    else:
        flabels = ['13150', '19450', '21825', '21925', '25775', '32000',
                   '36131.67', '36240', '36290', '38230.45', '44557.5', '51053.33',
                   '57550', '62175']
        plabels = ['Ks', 'Q']

    if fname is None:
        if multizone:
            fname = ['x_sampling.npy', 'y_sampling.npy']
        else:
            fname = ['input_mascaret.npy', 'output_mascaret.npy']
        fname = [os.path.join(PATH, p) for p in fname]

    sample = Sample(plabels=plabels, flabels=flabels, pformat='npy', fformat='npy')
    sample.read(space_fname=fname[0], data_fname=fname[1])
    sample.desc = desc
    return sample


def marthe():
    """MARTHE dataset."""
    desc = ("In 2005, CEA (France) and Kurchatov Institute (Russia) developed"
            " a model of strontium 90 migration in a porous water-saturated"
            " medium. The scenario concerned the temporary storage of"
            " radioactive waste (STDR) in a site close to Moscow. The main"
            " purpose was to predict the transport of 90Sr between 2002 and"
            " 2010, in order to determine the aquifer contamination. The"
            " numerical simulation of the 90Sr transport in the upper aquifer"
            " of the site was realized via the MARTHE code"
            " (developed by BRGM, France).")

    dataset = np.loadtxt(os.path.join(PATH, 'marthe.dat'), skiprows=1)

    plabels = ['per1', 'per2', 'per3', 'perz1', 'perz2', 'perz3', 'perz4',
               'd1', 'd2', 'd3', 'dt1', 'dt2', 'dt3', 'kd1', 'kd2', 'kd3',
               'poros', 'i1', 'i2', 'i3']

    flabels = ['p102K', 'p104', 'p106', 'p2.76', 'p29K',
               'p31K', 'p35K', 'p37K', 'p38', 'p4b']

    sample = Sample(plabels=plabels, flabels=flabels)
    sample += dataset
    sample.desc = desc
    return sample
