import zarr
import os

path = 'data_out/GSM7990532_slim3/region_Top/transcripts.zarr.zip'
store = zarr.ZipStore(path, mode='r')
root = zarr.open(store)

def print_tree(group, indent=''):
    for key in group.keys():
        print(f"{indent}{key} (type: {type(group[key])})")
        if isinstance(group[key], zarr.hierarchy.Group):
            print_tree(group[key], indent + '  ')

print_tree(root)
