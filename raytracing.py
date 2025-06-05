import sys
import tqdm
import numpy as np
import pandas as pd
import pyproj
import blockmodel
from obspy.taup import TauPyModel
from math import radians, degrees
from numpy import sin, cos, deg2rad
from obspy.geodetics import kilometer2degrees,degrees2kilometers

def output_xyz(array, filepath, output_v=True, input_in_depth=False):
    # Extract longitude, latitude, and depth from res[0]
    lons = array[:, 0]
    lats = array[:, 1]
    z = array[:, 2]
    if output_v: vals = array[:, 3]

    # Assume Earth radius in km
    R_earth = 6371.0

    # Convert latitude and longitude to radians
    lats_rad = deg2rad(lats)
    lons_rad = deg2rad(lons)

    # Compute radius at each depth (depth is in km)
    if input_in_depth: r = R_earth - z
    else: r = z

    # Convert to cartesian coordinates
    x = r * cos(lats_rad) * cos(lons_rad)
    y = r * cos(lats_rad) * sin(lons_rad)
    z = r * sin(lats_rad)
    # print(z.max(),z.min())

    # Normalize x, y, z to [-0.5, 0.5]
    x_norm = x / 2 / R_earth
    y_norm = y / 2 / R_earth
    z_norm = z / 2 / R_earth

    if output_v:
        df = pd.DataFrame(np.stack([x_norm, y_norm, z_norm, vals], axis=1), columns=["x", "y", "z", "val"])
    else:
        df = pd.DataFrame(np.stack([x_norm, y_norm, z_norm], axis=1), columns=["x", "y", "z"])
    df.to_csv(filepath, index=False)
    if verbose_level>2: print(df.to_csv(index=False))

def get_raypath_coordinates(lon1, lat1, lon2, lat2, source_depth_km, receiver_depth_km=0, phase_list=('P', 'S', 'ScS')):
    geod = pyproj.Geod(ellps="WGS84")
    velo = TauPyModel(model="prem")

    az, back_az, dist = geod.inv(lon1, lat1, lon2, lat2)
    distance_degree = kilometer2degrees(dist/1000)
    taup_arrivals = velo.get_ray_paths(source_depth_in_km=source_depth_km,
                                       distance_in_degree=distance_degree,
                                       phase_list=phase_list,
                                       receiver_depth_in_km=receiver_depth_km)
    if verbose_level>1: print(f"{len(taup_arrivals)} rays calculated for {', '.join(phase_list)} with distance {round(distance_degree,4)} deg.")
    
    # return taup_arrivals
    results = []
    taup_paths = [taup_arrival.path for taup_arrival in taup_arrivals]
    for taup_path in taup_paths:
        taup_dists = np.array([degrees2kilometers(degrees(row[2]))*1000 for row in taup_path])
        taup_depths = np.array([row[3] for row in taup_path])
        # print(az)
        lons, lats, azs = geod.fwd(lon1*np.ones_like(taup_dists), lat1*np.ones_like(taup_dists), az*np.ones_like(taup_dists), taup_dists)
        dz = kilometer2degrees(np.diff(taup_depths))
        dx = np.diff([degrees(row[2]) for row in taup_path])
        # return dx,dz
        p = 1/(degrees(1/taup_path[0][0])) #s/rad to s/deg
        p = 1/degrees2kilometers(1/p, 6371-taup_depths[1:]) #s/deg to s/km
        slownesses = p / np.sin(np.arctan2(dx,dz))
        # print(taup_path[0][0])
        slownesses = np.insert(slownesses,0,slownesses[0])
        results.append(np.array([lons, lats, taup_depths, slownesses]).T)
        
    return results

def euclidean_distance(point1, point2):
    """
    Calculate the straight-line (chord) distance between two points inside the Earth,
    given their longitude, latitude (in degrees), and depth (in km).
    """
    # Get polar coordinates
    lon1_in_deg = point1[0]
    lat1_in_deg = point1[1]
    dep1_in_km = point1[2]
    lon2_in_deg = point2[0]
    lat2_in_deg = point2[1]
    dep2_in_km = point2[2]

    # Earth's mean radius in km
    R_earth = 6371.0

    # Convert to spherical coordinates (radius from center, theta, phi)
    r1 = R_earth - dep1_in_km
    r2 = R_earth - dep2_in_km

    # Convert degrees to radians
    lon1_rad = np.deg2rad(lon1_in_deg)
    lat1_rad = np.deg2rad(lat1_in_deg)
    lon2_rad = np.deg2rad(lon2_in_deg)
    lat2_rad = np.deg2rad(lat2_in_deg)

    # Spherical to Cartesian
    x1 = r1 * np.cos(lat1_rad) * np.cos(lon1_rad)
    y1 = r1 * np.cos(lat1_rad) * np.sin(lon1_rad)
    z1 = r1 * np.sin(lat1_rad)

    x2 = r2 * np.cos(lat2_rad) * np.cos(lon2_rad)
    y2 = r2 * np.cos(lat2_rad) * np.sin(lon2_rad)
    z2 = r2 * np.sin(lat2_rad)

    # Euclidean distance
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

def geod_midpoint(point1, point2):
    # geod = pyproj.Geod(ellps="WGS84")
    # az, back_az, dist = geod.inv(point1[0], point1[1], point2[0], point2[1])
    # lon, lat, az = geod.fwd(point1[0], point1[1], az, dist/2)
    # return np.array([lon, lat, (point1[2]+point2[2])/2, (point1[3]+point2[3])/2])
    lon_diff = lambda lons: (lons[0]-lons[1]-360) if lons[0]-lons[1]>180 else (lons[0]-lons[1]+360) if lons[0]-lons[1]<-180 else lons[0]-lons[1]
    return point1 + np.array([lon_diff((point2[0],point1[0])), point2[1]-point1[1], point2[2]-point1[2], point2[3]-point1[3]]) / 2
    

def get_kernel_from_raypath(md: blockmodel.Model, raypath):
    epsilon = 1e-4
    row_kernel = np.zeros_like(md)
    lon = lambda phi: (phi + 180) % 360 - 180
    lat = lambda psi: (180 - psi) if psi > 90 else (-180 - psi) if psi < -90 else psi
    i = 0
    while(i < len(raypath)-1):
        slowness = lambda pt: pt[3]
        block = lambda pt: md.findBlocks(readable=True,
                                        find_one=True,
                                        lon=float(pt[0]),
                                        lat=float(pt[1]),
                                        dep=max(float(pt[2]), epsilon),
                                        )
        this_point = raypath[i]
        next_point_idx = i+1
        next_point = raypath[next_point_idx]
        this_block = block(this_point)
        next_block = block(next_point)
        if verbose_level>2: print(this_point[:3], this_block.id, next_point[:3], next_block.id, this_block == next_block)
        if this_block == next_block:
            row_kernel[this_block.id] -= euclidean_distance(this_point, next_point) * slowness(this_point)
            i += 1
        else:
            neighbor_blocks = md.findNeighbor(this_block, 'all')
            tmp_point = this_point
            while (not block(next_point) in neighbor_blocks):
                # mid_point = (tmp_point + next_point) / 2
                mid_point = geod_midpoint(tmp_point, next_point)
                if verbose_level>1: print("find neighbors...", block(this_point).id, block(next_point).id, "@", mid_point[:3])

                # if midpoint is neighbor, set as target point
                if block(mid_point) in neighbor_blocks:
                    next_point = mid_point
                    raypath = np.insert(raypath, i+1, mid_point, axis=0)
                    if verbose_level>1: print("insert neighbor:", mid_point[:3])

                # if midpoint and the working point belongs to the same block, move working point
                elif block(mid_point) == block(tmp_point):
                    tmp_point = mid_point

                # if midpoint and the target point belongs to the same block, move target point
                else:
                    next_point = mid_point
            
            # now the target point is truly a neighbor
            next_block = block(next_point)

            if i+1 == len(raypath):
                boundary_point = next_point
            else:
                lon_diff = lambda lons: (lons[0]-lons[1]-360) if lons[0]-lons[1]>180 else (lons[0]-lons[1]+360) if lons[0]-lons[1]<-180 else lons[0]-lons[1]
                point_by_ratio = lambda r: this_point + np.array([lon_diff((next_point[0],this_point[0])), next_point[1]-this_point[1], next_point[2]-this_point[2], next_point[3]-this_point[3]]) * r
                if next_block in md.findNeighbor(this_block, 'E'):
                    boundary_point = point_by_ratio(lon_diff((lon(this_block.east),this_point[0]))/lon_diff((next_point[0],this_point[0])))
                    direction = [1, 0, 0, 0]
                elif next_block in md.findNeighbor(this_block, 'W'):
                    boundary_point = point_by_ratio(lon_diff((lon(this_block.west),this_point[0]))/lon_diff((next_point[0],this_point[0])))
                    direction = [-1, 0, 0, 0]
                elif next_block in md.findNeighbor(this_block, 'N'):
                    boundary_point = point_by_ratio((90-this_block.north - this_point[1])/(next_point[1] - this_point[1]))
                    direction = [0, 1, 0, 0]
                elif next_block in md.findNeighbor(this_block, 'S'):
                    boundary_point = point_by_ratio((90-this_block.south - this_point[1])/(next_point[1] - this_point[1]))
                    direction = [0, -1, 0, 0]
                elif next_block in md.findNeighbor(this_block, 'D'):
                    boundary_point = point_by_ratio((6371.-this_block.bottom - this_point[2])/(next_point[2] - this_point[2]))
                    direction = [0, 0, 1, 0]
                elif next_block in md.findNeighbor(this_block, 'U'):
                    boundary_point = point_by_ratio((6371.-this_block.top - this_point[2])/(next_point[2] - this_point[2]))
                    direction = [0, 0, -1, 0]
                
                if verbose_level>1: print("neighbor:", direction)
                boundary_point_across = boundary_point + np.array(direction) * epsilon
                boundary_point_across[0] = lon(boundary_point_across[0])
                boundary_point_across[1] = lat(boundary_point_across[1])
                # print(6371.-this_block.bottom, this_point[2], next_point[2])
                # print((6371.-this_block.bottom - this_point[2])/(next_point[2] - this_point[2]))

            row_kernel[this_block.id] -= euclidean_distance(this_point, boundary_point) * slowness(this_point)
            if verbose_level>1: print("insert:", boundary_point_across[:3])
            raypath = np.insert(raypath, i+1, boundary_point_across, axis=0)
            i += 1
    
    return row_kernel

if __name__ == '__main__':
    # Default verbose level
    verbose_level = 0

    # Parse command line arguments for verbosity flags
    for arg in sys.argv[1:]:
        if arg == '-v':
            verbose_level = max(verbose_level, 1)
        elif arg == '-vv':
            verbose_level = max(verbose_level, 2)
        elif arg == '-vvv':
            verbose_level = max(verbose_level, 3)

    # Initialize model
    grid = blockmodel.Model()
    velocity = TauPyModel(model="prem")
    
    if verbose_level>0: print("loading pick catalog...")
    df = pd.read_pickle('globocat_1.2.2_sample.pkl')
    res = [ get_raypath_coordinates(row['origin_lon'],row['origin_lat'],row['station_lon'],row['station_lat'],row['origin_dep']) for ind, row in df.sample(1000).iterrows() ]
    print("pick catalog loaded.")

    # Generate P kernel
    kernel = np.array([get_kernel_from_raypath(grid, path[2]) for path in tqdm.tqdm(res, desc="Generate ScS kernel")])

    kernel_sparse = []
    for i, val in enumerate(np.sum(kernel, axis=0)):
        if val != 0: 
            kernel_sparse.append([grid[i].clon-360 if grid[i].clon>180 else grid[i].clon, 90-grid[i].clat, grid[i].crad, val])
    kernel_sparse = np.array(kernel_sparse)

    output_xyz(kernel_sparse, 'globocat_1.2.2_sample_ScS_fast_sparse.xyz', output_v=True, input_in_depth=False)
