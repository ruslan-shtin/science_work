import os, glob, sys, math, itertools, operator

varList = [ 'Cp', 'M_is', ]

path_to_data = r'Y:\old_calculators\old_calculators\EADT\model_test_SM4_FLEX\mesh0'

data_dirs= glob.glob(os.path.join(path_to_data,'Def*','00*'))
print(data_dirs)
try: paraview.simple
except: from paraview.simple import *

for ddir in data_dirs:
    cgnsFile  = os.path.join(ddir,'wing_fam_nodes.cgns')
    
    outFile = os.path.split(ddir)[-1]+'.csv'
    
    multiblock_field = CGNSSeriesReader( FileNames = [ cgnsFile, ] )
    multiblock_field.PointArrayStatus = varList

    merged_field = MergeBlocks( Input = multiblock_field )
    
    SaveData( outFile , proxy = merged_field, Precision = 12 ) #, ChooseArraysToWrite=1, PointDataArrays=['Cp', 'M_is'], FieldAssociation='Points' )
    Delete( multiblock_field )
    Delete( merged_field     )
