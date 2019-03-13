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

Guide for installing needed python packages
1. Download latest anaconda here: https://www.anaconda.com/download/#linux
2. Copy the downloaded file (AnacondaXXX) to your home folder on the cluster (from windows you can do this using WinSCP)
3. Install it using $ bash AnacondaXXX
4. Add to path $ export PATH=~/anaconda3/bin:$PATH
5. Update conda update -n base -c defaults conda
6. Permanently add conda to path: in the home directory execute $  echo 'export PATH=~/anaconda3/bin:$PATH' >> ~/.bashrc

Install pyopencl: (might not be needed, has been added to setup.py, but check it's installed)
1. $ conda install -n tomography -c conda-forge pyopencl

To permanently add something to PATH ect. use echo "export PATH=$PATH:/path/to/dir" >> /home/KUID/.bash_profile

1. echo 'export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH' >> ~/.bashrc
2. echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
3. echo 'export CPATH=/usr/local/cuda/include:$CPATH' >> ~/.bashrc
4. echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc

To see ~/.bashrc file and check things are ok, write $ vi ~/.bashrc. To edit type c. To exit with no changes :q!+enter, to exit and save :x+enter, to see all environment variables $ env
To find stuff:  find / -iname filename  2>/dev/null

The full dataset can be found in the dev branch in the folder zipped data. It is a split archive, extract with the command $ zip -F source-data.zip --out tmp.zip && unzip tmp.zip && rm tmp.zip
or use the bash script
