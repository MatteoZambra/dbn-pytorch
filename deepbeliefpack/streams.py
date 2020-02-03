

import os

def check_create_directory(path):
    
    if not os.path.exists(path):
        print('creating directory')
        os.system('mk_dir ' + path)
    #end
#end