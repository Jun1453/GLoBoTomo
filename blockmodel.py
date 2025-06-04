import numpy as np

# RADIUS1 = [3484.3,3661.0,3861.0,4061.0,4261.0,4461.0,4861.0,5061.0,5261.0,5411.0,5561.0,5711.0,5841.0,5971.0,6071.0,6171.0,6260.0,6349.0]
RADIUS1 = [3480,3661.0,3861.0,4061.0,4261.0,4461.0,4861.0,5061.0,5261.0,5411.0,5561.0,5711.0,5841.0,5971.0,6071.0,6171.0,6260.0,6349.0] # for PREM
RADIUS2 = [3661.0,3861.0,4061.0,4261.0,4461.0,4861.0,5061.0,5261.0,5411.0,5561.0,5711.0,5841.0,5971.0,6071.0,6171.0,6260.0,6349.0,6371.0]

class Model(list):
    def __init__(self, bsize=4, nshell=18):
        self.number_of_latitude_bands = int(np.round(180 / bsize))
        self.block_size = 180 / self.number_of_latitude_bands
        self.number_of_shells = nshell
        self.number_of_blocks_in_band = np.zeros(self.number_of_latitude_bands)
        block_count = 0
        for shell in range(self.number_of_shells):
            bottom_radius = RADIUS1[shell]
            top_radius = RADIUS2[shell]
            for latitude_band in range(self.number_of_latitude_bands):
                band_center_latitude = (latitude_band + 0.5) * self.block_size
                shrinked_lontitude = np.sin(np.deg2rad(band_center_latitude))
                self.number_of_blocks_in_band[int(latitude_band)] = max(1, np.round(360 / self.block_size * shrinked_lontitude))
                width_on_band = 360 / self.number_of_blocks_in_band[int(latitude_band)]
                for block_order in range(int(self.number_of_blocks_in_band[int(latitude_band)])):
                    block_center_longitude = (block_order + 0.5) * width_on_band
                    self.append(Block(bid=block_count, context=self, clon=block_center_longitude, clat=band_center_latitude, crad=np.mean([bottom_radius, top_radius]), hx=width_on_band, hy=self.block_size, hz=top_radius-bottom_radius))
                    block_count += 1

    def getKernel(self, raypaths):
        kernel = np.zeros((len(raypaths), len(self)))
        for raypath_index, raypath in enumerate(raypaths):
            for block in self:
                kernel[raypath_index, block.id] += 1
        return kernel

    def __repr__(self):
        return f'A model contains {len(self)} blocks.'
    def findNeighbor(self, block, direction):
        if not direction in ['N', 'S', 'E', 'W', 'U', 'D', 'all']:
            raise ValueError(f'Invalid direction: {direction}')
        condition = {'rad': block.crad, 'lat': block.clat, 'lon': block.clon}
        if direction == 'N' or direction == 'S':
            dy = block.south - block.north
            dy *= -1 if direction == 'N' else 1
            condition['lat'] += dy
            results = []
            condition_backup = condition
            expanded_lon_conditions = np.append(np.arange(block.west+1e-6,block.east-1e-6,180/self.number_of_latitude_bands/1.8), block.east-1e-6)
            lon = lambda x: x - 360 if x > 180 else x
            print([lon(i) for i in expanded_lon_conditions])
            for lon_cond in expanded_lon_conditions:
                condition = condition_backup
                condition['lon'] = lon_cond
                if condition['lat'] > 180:
                    condition['lat'] = 360 - condition['lat']
                    condition['lon'] = (lon_cond + 180) % 360
                elif condition['lat'] < 0:
                    condition['lat'] *= -1
                    condition['lon'] = (lon_cond + 180) % 360
                result = self.findBlocks(readable=False, find_one=True, **condition)
                if not result in results:
                    results.append(result)
            return results
        elif direction == 'E' or direction == 'W':
            dx = block.east - block.west
            dx *= -1 if direction == 'W' else 1
            condition['lon'] = (condition['lon'] + dx) % 360
        elif direction == 'D':
            condition['rad'] = block.bottom - 1
        elif direction == 'U':
            condition['rad'] = block.top + 1
        elif direction == 'all':
            # flatten nested list
            neighbors = [self.findNeighbor(block, direction) for direction in ['N', 'S', 'E', 'W', 'U', 'D']]
            # Flatten the list if any element is itself a list
            flat_neighbors = []
            for n in neighbors:
                if isinstance(n, list):
                    flat_neighbors.extend(n)
                else:
                    flat_neighbors.append(n)
            return flat_neighbors
            
        return self.findBlocks(readable=False, find_one=False, **condition)
        
    def findBlocks(self, readable=True, find_one=False, **kwargs):
        rad = lambda z: 6371 - z
        lat = lambda y: (90 - y) if readable is True else y
        lon = lambda x: (x + 360 if x < 0 else x) if readable is True else x

        keys = kwargs.keys()
        dep_val = rad(kwargs['dep']) if 'dep' in keys else None
        rad_val = kwargs['rad'] if 'rad' in keys else None
        lat_val = lat(kwargs['lat']) if 'lat' in keys else None
        lon_val = lon(kwargs['lon']) if 'lon' in keys else None

        # Pre-round lon/lat if needed for efficiency
        if lon_val is not None:
            lon_val_rounded = np.round(lon_val, 4)

        results = []
        for block in self:
            if dep_val is not None:
                if dep_val >= block.top or dep_val < block.bottom:
                    continue
            if rad_val is not None:
                if rad_val >= block.top or rad_val < block.bottom:
                    continue
            if lat_val is not None:
                if lat_val >= block.south or lat_val < block.north:
                    continue
            if lon_val is not None:
                block_east = np.round(block.east, 4)
                block_west = np.round(block.west, 4)
                if lon_val_rounded >= block_east or lon_val_rounded < block_west:
                    continue
            if find_one:
                return block
            results.append(block)
        return results



# Block objects record geometry for a kernel matrix
class Block():
    def __init__(self, bid, context, clon, clat, crad, hx, hy, hz):
        self.id = bid
        self.context = context
        (self.clon, self.clat, self.crad) = (clon, clat, crad)
        (self.width, self.length, self.thickness) = (hx, hy, hz)
        (self.west, self.east) = ((clon - 1/2 * hx), (clon + 1/2 * hx))
        (self.north, self.south) = ((clat - 1/2 * hy), (clat + 1/2 * hy))
        (self.bottom, self.top) = ((crad - 1/2 * hz), (crad + 1/2 * hz))
    def __repr__(self):
        intro = "### Block information ###\n"
        for key, value in self.readable().items():
            intro += f'# {key}: {value}\n'
        intro += "########################\n"
        return intro
    def neighbor(self, direction):
        return self.context.findNeighbor(self, direction)
    def readable(self):
        dep = lambda r: 6371 - r
        lat = lambda y: 90 - y
        lon = lambda x: x - 360 if x > 180 else x
        return {'id': self.id,
                'center_depth': dep(self.crad),
                'deepest': dep(self.bottom),
                'shallowest': dep(self.top),
                'center_radius': self.crad,
                'top': self.top,
                'bottom': self.bottom,
                'latitude': lat(self.clat),
                'north':lat(self.north),
                'south':lat(self.south),
                'longitude': lon(self.clon),
                'west': lon(self.west),
                'east': lon(self.east)}