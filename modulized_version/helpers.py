import time, math, gmplot, googlemaps, random
import numpy as np
import pandas as pd
import torch

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

#def nearest_roads(path):
#    gmaps = googlemaps.Client(key='AIzaSyBdWT6RiC_PwypB-PBWcn0aK6kBo_U5VQ4')

def random_training_Var(pathes):
    while True:
        path = random.choice(list(pathes.values()))
        if len(path) >= 2:
            break
    inp = torch.from_numpy(path[:-1]).float()
    tar = torch.from_numpy(path[1:]).float()
    return Variable(inp).contiguous(), Variable(tar).contiguous()

def random_path(pathes):
    while True:
        path = random.choice(list(pathes.values()))
        if len(path) >= 2:
            break
    path = path[:,[2,1]]
    if len(path) > 100:
        gap = int(len(path) / 100.0)+1
        l = int(len(path) / gap)
        index = np.array(range(0,l))*gap
        path =  path[index]
    return path

def map_plot(path):
    lats, lngs = path[:,0], path[:,1]
    gmap = gmplot.GoogleMapPlotter(lats[0],lngs[0], 16)
    gmap.scatter(lats, lngs, '#3B0B39', size=7, marker=False)
    gmap.draw("mymap.html")            

def load_data():
    return pd.read_csv('../../data/data_20161117/gps_20161117',names = ['driver','order','time','lng','lat'])
    
def load_pathes():
    df_gps = load_data()
    grouped = df_gps.groupby('order')
    pathes = {}
    i = 0
    for x in grouped:
        i+=1
        if i%20000 == 0:
            print(i)
        order = x[0]
        mt = x[1][['time','lat','lng']].as_matrix()
        mt[:,0] -= mt[0,0] 
        pathes[x[0]] = mt
    return pathes

def to_var(triple):
    tensor = torch.from_numpy(triple).float()
    return Variable(tensor)

def to_road(path):
    gmaps = googlemaps.Client(key='AIzaSyB8VPi0g-OLtuhUwZno2c6LDWFglsmkCd8')
    road = []
    gps = []
    for x in gmaps.snap_to_roads(path):
        rd = x['placeId']
        if not road or rd != road[-1]:
            road.append(rd)
            gps.append( (x['location']['latitude'], x['location']['longitude']) )
    return np.array(road), np.array(gps)

def to_road_fixed_gps(path):
    gmaps = googlemaps.Client(key='AIzaSyB8VPi0g-OLtuhUwZno2c6LDWFglsmkCd8')
    road = []
    gps = []
    for x in gmaps.nearest_roads(path):
        rd = x['placeId']
        if not road or rd != road[-1]:
            road.append(rd)
            gps.append( (x['location']['latitude'], x['location']['longitude']) )
    return np.array(road), np.array(gps)

def sparse(path):
    l = len(path)
    if l <= 100:
        return path
    else:
        gap = int(l / 100) + 1
        idx = np.array(range(0,len(path), gap))
    return path[idx,:]

def street_name(placeID):
    gmaps = googlemaps.Client(key='AIzaSyBGYI1qNfrVF2AdrRw8LFRuc2Z2KU2-06Q')
    print(placeID)
    try:
        return gmaps.place(placeID)['result']['address_components'][1]['long_name']
    except:
        import pdb; pdb.set_trace()
