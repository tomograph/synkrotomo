Synkrotomo setup guide linux

Guide for installing futhark:
1. Download the latest tarball here: https://futhark-lang.org/releases/futhark-nightly-linux-x86_64.tar.xz
2. Unpack it using the command $ tar -xf futhark-nightly-linux-x86_64.tar.xz
4. Then go to the folder where it unpacked $ cd futhark-nightly-linux-x86_64
5. Execute the install command $ make install

Download synkrotomo repository
1. Download https://github.com/tomograph/synkrotomo

Distribute and install synkrotomo
1. $ cd synkrotomo
2. $ make lib
3. $ python setup.py sdist
4. copy the generated tarball in the dist folder to your home folder
5. Unpack the generated tarball in the dist folder $ tar -xf synkrotomo-xx.tar.gz
6. cd to the directory synkrotomo-xx
7. install using $python setup.py install

You should now be able to import the different algorithms in python using "from futhark import SIRT" and calling them like this:

sirt = SIRT.SIRT()
result = sirt.main(theta.astype(np.float32), rhozero, deltarho, emptyimage, sinogram.flatten().astype(np.float32), iterations).get()
reconstruction_result = result.reshape((size,size))
