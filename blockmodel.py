import numpy as np

RADIUS1 = [3484.3,3661.0,3861.0,4061.0,4261.0,4461.0,4861.0,5061.0,5261.0,5411.0,5561.0,5711.0,5841.0,5971.0,6071.0,6171.0,6260.0,6349.0]
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


    def __repr__(self):
        return f'A model contains {len(self)} blocks.'
    def findNeighbor(self, block, direction):
        if not direction in ['N', 'S', 'E', 'W', 'U', 'D']:
            return None
        condition = {'rad': block.crad, 'lat': block.clat, 'lon': block.clon}
        if direction == 'N' or direction == 'S':
            dy = block.south - block.north
            dy *= -1 if direction == 'S' else 1
            condition['lat'] += dy
            if condition['lat'] > 180:
                condition['lat'] = 360 - condition['lat']
                condition['lon'] = (condition['lon'] + 180) % 360
            elif condition['lat'] < 0:
                condition['lat'] *= -1
                condition['lon'] = (condition['lon'] + 180) % 360
        elif direction == 'E' or direction == 'W':
            dx = block.east - block.west
            dx *= -1 if direction == 'W' else 1
            condition['lon'] = (condition['lon'] + dx) % 360
        else:
            dz = block.top - block.bottom
            dz *= -1 if direction == 'D' else 1
            condition['rad'] += dz
            
        results = self.findBlocks(readable=False, find_one=True, **condition)
        return results[0] if len(results) == 1 else None
        
    def findBlocks(self, readable=True, find_one=False, **kwargs):
        rad = lambda z: 6371 - z
        lat = lambda y: np.round(90 - y) if readable else y
        lon = lambda x: (x + 360 if x < 0 else x) if readable else x

        keys = kwargs.keys()
        results = []
        for block_index in range(len(self)):
            block = self[block_index]
            if 'dep' in keys:
                if rad(kwargs['dep']) >= block.top or rad(kwargs['dep']) < block.bottom:
                    continue
            if 'rad' in keys:
                if kwargs['rad'] >= block.top or kwargs['rad'] < block.bottom:
                    continue
            if 'lat' in keys:
                if lat(kwargs['lat']) >= block.south or lat(kwargs['lat']) < block.north:
                    continue
            if 'lon' in keys:
                if np.round(lon(kwargs['lon']), 4) >= np.round(block.east, 4) or np.round(lon(kwargs['lon']), 4) < np.round(block.west, 4):
                    continue
            results.append(self[block_index])
            if find_one:
                return results
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