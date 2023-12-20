import numpy as np
import pylab as pl

from constants import GHz_2pi

def readData(fname):
    d = np.load(fname)
    #for k in d.keys(): print (k)
    #for k in ['a', 'b', 's', 'd']:
    #    if k in d.keys(): print (k, " = ", d[k])
    omega = d['omega']
    T = d['T1_fw']
    R = d['R1_fw']
    k = 0.0 * omega
    #if 'k' in d.keys():
    k = d['k_fw'] #0.0 * omega + 0.0
    k_bk = 0.0 * omega + 0.0
    err = d['err_fw'] #0.0 * np.abs(T)
    T_bk = d['T1_bk']
    R_bk = d['R1_bk']
    #err_bk = 0.0 * np.abs(T)
    err_bk = d['err_bk']
    k_bk = d['k_bk']
    k_orig = d['k']

    result = []
    for i in range(len(omega)):
        item = dict(
            omega = omega[i], k_orig = k[i], k = k[i], k_bk = k_bk[i],
            err = err[i], err_bk = err_bk[i],
            T = T[i], T_bk = T_bk[i], R = R[i], R_bk = R_bk[i],
            fname = fname)
        result.append(item)
    return result


def joinData(fnames):
    result = []
    for fname in fnames:
        dataset = readData(fname)
        result.extend(dataset)
    result.sort(key = lambda item: item['omega'])
    res_filtered = []
    last_group = []
    for item in result:
        print ("item", item['omega'] / GHz_2pi, item['fname'])
        if len(last_group) == 0:
            last_group.append(item)
            print ('start new group')
        elif abs(last_group[-1]['omega'] - item['omega']) < 1e-6:
            last_group.append(item)
            print ('add to the group')
        else:
            last_group.sort(key = lambda item: item['err'])
            res_filtered.append(last_group[0])
            last_group = []
            print('add to the filtered result')
    if len(last_group):
        last_group.sort(key = lambda item: item['err'])
        res_filtered.append(last_group[0])
        print('flush the group')
    print ("total: ", len(res_filtered))
    res_data = dict()
    keys_to_copy = [
        'omega', 'T', 'T_bk', 'R', 'R_bk',
        'k_orig', 'k', 'k_bk', 'err', 'err_bk'
    ]
    for k in keys_to_copy:
        res_data[k] = np.array([t[k] for t in res_filtered])
    print ("res_data:", res_data)
    return res_data


def cmpData(fname, data):
    d2 = np.load(fname)
    print ("data: ")
    for k in data.keys(): print (k)
    print ("d2: ")
    for k in d2.keys(): print (k)

    omega1 = data['omega']
    omega2 = d2['omega']
    T1_fw = data['T']
    T1_bk = data['T_bk']
    R1_fw = data['R']
    R1_bk = data['R_bk']
    print ("shapes: ", np.shape(T1_fw), np.shape(T1_bk),
           np.shape(R1_fw), np.shape(R1_bk))
    T2_fw = d2['T1_nrf']
    T2_bk = d2['T2_nrf']
    R2_fw = d2['R1_nrf']
    R2_bk = d2['R2_nrf']

    omega2 = omega2[:len(T2_fw)]
    omega1 = omega1[:len(T1_fw)]

    o_min = 3.5 * GHz_2pi
    o_max = 4.0 * GHz_2pi

    i1_filt = [t for t in range(len(omega1)) if omega1[t] > o_min
               and omega1[t] < o_max]
    i2_filt = [t for t in range(len(omega2)) if omega2[t] > o_min
               and omega2[t] < o_max]

    T1_filt = np.array(np.abs([T1_bk[t] for t in i1_filt]))
    o1_filt = np.array(([omega1[t] for t in i1_filt]))
    T2_filt = np.array(np.abs([T2_bk[t] for t in i2_filt]))
    o2_filt = np.array(([omega2[t] for t in i2_filt]))

    i1_min = np.argmin(np.abs(T1_filt))
    o1_min = o1_filt[i1_min]
    i2_min = np.argmin(np.abs(T2_filt))
    o2_min = o2_filt[i2_min]

    do = o2_min - o1_min

    if 'T1' in d2.keys():
       T2_fw_iso = d2['T1']
       T2_bk_iso = d2['T2']
       R2_fw_iso = d2['R1']
       R2_bk_iso = d2['R2']

    pl.figure()
    pl.xlabel (r'Frequency $\omega/2\pi$, $GHz$')
    pl.ylabel(r'Transmissivity $|T(\omega)|$')
    p1 = pl.plot(omega1 / GHz_2pi, np.abs(T1_fw), label='T ->, exact')
    p2 = pl.plot(omega1 / GHz_2pi, np.abs(T1_bk), label='T <-  ')
    if 'T1' in d2.keys():
       pl.plot(omega2 / GHz_2pi, np.abs(T2_fw), '--', color=p1[0].get_color(),
            label='T -> , nr-field')
       pl.plot(omega2 / GHz_2pi, np.abs(T2_bk), '--', color=p2[0].get_color(),
            label='T <- , nr-field')
    else:
        #pl.plot(omega2 / GHz_2pi, np.abs(T2_fw), '--', color=p1[0].get_color(),
        #    label='T -> , with self-energy')
        #pl.plot(omega2 / GHz_2pi, np.abs(T2_bk), '--', color=p2[0].get_color(),
        #    label='T <- , with self-energy')
        pl.plot(omega2 / GHz_2pi, np.abs(T2_fw), '--', color=p1[0].get_color(),
            label='T -> , with off-resonant terms')
        pl.plot(omega2 / GHz_2pi, np.abs(T2_bk), '--', color=p2[0].get_color(),
            label='T <- , with off-resonant terms')
       
    if False or 'T1' in d2.keys():
       pl.plot((omega2 - do)/ GHz_2pi, np.abs(T2_fw), '.', ms=2.0, color=p1[0].get_color(),
            label='T -> , nr-field shifted')
       pl.plot((omega2 - do) / GHz_2pi, np.abs(T2_bk), '.', ms=2.0, color=p2[0].get_color(),
            label='T <- , nr-field shifted')
    if 'T1' in d2.keys():
      pl.plot(omega2 / GHz_2pi, np.abs(T2_fw_iso), linestyle='dashdot', color=p1[0].get_color(),
            label='T -> , isolated')
      pl.plot(omega2 / GHz_2pi, np.abs(T2_bk_iso), linestyle='dashdot', color=p2[0].get_color(),
            label='T <- , isolated')

    pl.legend()
    
    print ("shapes: ", np.shape(T1_fw), np.shape(T1_bk),
           np.shape(R1_fw), np.shape(R1_bk), np.shape(omega1))
    pl.figure()
    pl.xlabel (r'Frequency $\omega/2\pi$, $GHz$')
    pl.ylabel(r'Reflectivity $|R(\omega)|$')
    p1 = pl.plot(omega1 / GHz_2pi, np.abs(R1_fw), label='R ->, exact')
    p2 = pl.plot(omega1 / GHz_2pi, np.abs(R1_bk), label='R <-  ')
    if 'T1' in d2.keys():
       pl.plot(omega2 / GHz_2pi, np.abs(R2_fw), '--', color=p1[0].get_color(),
            label='R -> , nr-field')
       pl.plot(omega2 / GHz_2pi, np.abs(R2_bk), '--', color=p2[0].get_color(),
            label='R <- , nr-field')
    else:
       #pl.plot(omega2 / GHz_2pi, np.abs(R2_fw), '--', color=p1[0].get_color(),
       #     label='R -> , with self-energy')
       #pl.plot(omega2 / GHz_2pi, np.abs(R2_bk), '--', color=p2[0].get_color(),
       #     label='R <- , with self-energy')
       pl.plot(omega2 / GHz_2pi, np.abs(R2_fw), '--', color=p1[0].get_color(),
            label='R -> , with off-resonant terms')
       pl.plot(omega2 / GHz_2pi, np.abs(R2_bk), '--', color=p2[0].get_color(),
            label='R <- , with off-resonant terms')
    if 'T1' in d2.keys():
       pl.plot((omega2 -do) / GHz_2pi, np.abs(R2_fw), '.', ms=2.0, color=p1[0].get_color(),
            label='R -> , nr-field shfited')
       pl.plot((omega2 - do) / GHz_2pi, np.abs(R2_bk), '.', ms=2.0, color=p2[0].get_color(),
            label='R <- , nr-field shifted')
    if 'T1' in d2.keys():
      pl.plot(omega2 / GHz_2pi, np.abs(R2_fw_iso), linestyle='dashdot', color=p1[0].get_color(),
            label='R -> , isolated')
      pl.plot(omega2 / GHz_2pi, np.abs(R2_bk_iso), linestyle='dashdot', color=p2[0].get_color(),
            label='R <- , isolated')

    pl.legend()

    pl.figure()
    p0 = pl.plot(omega1 / GHz_2pi, R1_fw.real, label='Re R_fw')
    p1 = pl.plot(omega1 / GHz_2pi, R1_fw.imag, label='Im R_fw')
    p2 = pl.plot(omega1 / GHz_2pi, R1_bk.real, label='Re R_bk')
    p3 = pl.plot(omega1 / GHz_2pi, R1_bk.imag, label='Im R_bk')
    pl.plot(omega2 / GHz_2pi, R2_fw.real, '--', label='Re R_fw',
            color = p0[0].get_color())
    pl.plot(omega2 / GHz_2pi, R2_fw.imag, '--', label='Im R_fw',
            color = p1[0].get_color())
    pl.plot(omega2 / GHz_2pi, R2_bk.real, '--', label='Re R_bk',
            color = p2[0].get_color())
    pl.plot(omega2 / GHz_2pi, R2_bk.imag, '--', label='Im R_bk',
            color = p3[0].get_color())
    pl.legend()
    pl.figure()
    p0 = pl.plot(omega1 / GHz_2pi, T1_fw.real, label='Re T_fw')
    p1 = pl.plot(omega1 / GHz_2pi, T1_fw.imag, label='Im T_fw')
    p2 = pl.plot(omega1 / GHz_2pi, T1_bk.real, label='Re T_bk')
    p3 = pl.plot(omega1 / GHz_2pi, T1_bk.imag, label='Im T_bk')
    pl.plot(omega2 / GHz_2pi, T2_fw.real, '--', label='Re T_fw',
            color = p0[0].get_color())
    pl.plot(omega2 / GHz_2pi, T2_fw.imag, '--', label='Im T_fw',
            color = p1[0].get_color())
    pl.plot(omega2 / GHz_2pi, T2_bk.real, '--', label='Re T_bk',
            color = p2[0].get_color())
    pl.plot(omega2 / GHz_2pi, T2_bk.imag, '--', label='Im T_bk',
            color = p3[0].get_color())
    pl.legend()
    pl.show()
    

import sys
data = joinData(sys.argv[2:])
cmpData(sys.argv[1], data)
