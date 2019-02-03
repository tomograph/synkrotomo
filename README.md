Synkrotomo setup guide for Image cluster

First you need access to the cluster. Follow this guide:
http://image.diku.dk/mediawiki/index.php/Slurm_Cluster

Guide for installing futhark:
1. Download the latest tarball here: https://futhark-lang.org/releases/futhark-nightly-linux-x86_64.tar.xz
2. Copy it to your home folder on the cluster (from windows you can do this using WinSCP)
3. Unpack it using the command $ tar -xf futhark-nightly-linux-x86_64.tar.xz
4. Then go to the folder where it unpacked $ cd futhark-nightly-linux-x86_64
5. Execute the install command $ PREFIX=$HOME/.local make install

Download synkrotomo repository
1. Download https://github.com/tomograph/synkrotomo
2. Copy it to your home folder on the cluster (from windows you can do this using WinSCP)

Distribute and install synkrotomo
1. $ cd synkrotomo
2. $ make lib
3. $ python setup.py sdist
4. copy the generated tarball in the dist folder to your home folder
5. Unpack it using the command $ tar -xf synkrotomo-xx.tar.gz
6. cd to the directory synkrotomo-xx
7. enable tomography environment using $source activate tomography
8. install using $python setup.py install
9. Deactivate the tomography environment if you don't need it anymore $ source deactivate tomography
10. Remember to activate it when you want to use it

Install pyopencl: (might not be needed, has been added to setup.py, but check it's installed)
1. $ conda install -n tomography -c conda-forge pyopencl
