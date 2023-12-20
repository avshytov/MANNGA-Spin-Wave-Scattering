import constants

def print_items_with_prio(items):
    items = list(items)
    items.sort(key = lambda item: item['prio'])
    for item in items:
        print (item['key'], item['value'], item['unit'], type(item['value']))
        try:
            print(item['key'], '=', item['value'] / item['unit'],
              item['unit_name'])
        except:
            print (item['key'], ':', item['value'])
    
    
def print_config(d):
    import constants
    res_items = []
    for k in d.keys():
        if k[0:4] == 'res_':
            if k[-3:] == '_Ms':
                res_items.append(dict(prio=0, key=k, value=d[k], 
                                  unit=constants.kA_m, unit_name='kA_m'))
            elif k[-6:] == '_Bbias':
                res_items.append(dict(prio=1, key=k, value=d[k], 
                                  unit=constants.mT, unit_name='mT'))
            elif (k[-2:] == '_W') or (k[-2:] == '_L')  or (k[-2:]== '_H'):
                res_items.append(dict(prio=2, key=k, value=d[k], 
                                  unit=constants.nm, unit_name='nm'))
            else:
                res_items.append(dict(prio=1000, key=k, value=d[k], 
                                  unit=1.0, unit_name=''))

    print_items_with_prio(res_items)
    print ('z_res = ', d['z_res'] / constants.nm, 'nm')
    
    slab_items = []
    for k in d.keys():
        if k[0:5] == 'slab_':
            unit = (1.0, '')
            if k[-3:] == '_Ms':
                slab_items.append(dict(prio=0, key=k, value=d[k],
                                       unit=constants.kA_m, unit_name='kA/m'))
                unit = (constants.kA_m, 'kA/m')
            elif k[-5:] == '_Bext':
                slab_items.append(dict(prio=1, key=k, value=d[k],
                                       unit=constants.mT, unit_name='mT'))
            elif (k[-2:] == '_d') or (k[-2:] == '_a') or (k[-2:] == '_b'):
                slab_items.append(dict(prio=2, key=k, value=d[k],
                                       unit=constants.nm, unit_name='nm'))
            else:    
                slab_items.append(dict(prio=1000, key=k, value=d[k],
                                       unit=1.0, unit_name=''))
    print_items_with_prio(slab_items)
