# import libraries
import os
import sys

def main():

  # set the directory
  dataset = sys.argv[-1]
  root = '/home/nasrim/data/raw_data/'+dataset+'/'
  os.makedirs(root, exist_ok = True)
  os.chdir(root)
   
  if dataset == 'tdrive':

    for i in range(6,15):
      os.system('curl https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/0'+str(i)+'.zip -o td'+str(i)+'.zip')
      os.system('unzip '+ str(root)+'td'+str(i)+'.zip')
      print('Dataset '+str(i)+' downloaded!', flush=True)
      print('Dataset '+dataset+' Downloaded!', flush=True)
      os.system('rm -r *.zip') 
   
  if dataset == 'geolife':
    os.system('curl https://download.microsoft.com/download/F/4/8/F4894AA5-FDBC-481E-9285-D5F8C4C4F039/Geolife%20Trajectories%201.3.zip -o geolife.zip')
    os.system('unzip geolife.zip')
    os.system('rm -r *.zip')

if __name__=='__main__':
  main()

