import os, glob, shutil


# computationPath = os.getcwd()
computationPath = os.path.normpath(r'\\sm-server\Data\old_calculators\old_calculators\EADT\model_test_SM4_FLEX\mesh0')
pattern = os.path.join(computationPath, 'Deformed*')
folders = sorted([os.path.split(item)[-1] for item in glob.glob(pattern) if os.path.isdir(item)])
res_file = os.path.join('results', 'z_post_proc.zresult')

###############################################



# try:
# dst_file = os.path.join(os.getcwd(), mesh, 'results.txt')
# with open(dst_file, 'w') as fo:
#     fo.write(header)

for folder in folders:
  regime = folder[9:]
  print(regime)
  src_folder = os.path.join(computationPath, folder, regime)
  src_file = os.path.join(src_folder, res_file)
  dst_file = regime+'.zresult'
  shutil.copyfile(src_file, dst_file)

# except Exception as err:
#     print(err)


