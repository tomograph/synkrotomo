import numpy as np
import re
import tomo_lib
# with open("output//sirtresult.out") as f:
#     content = f.readlines()
# content = [x.strip() for x in content]
# content = content[0].split("[")
# content = [re.sub('\], ', '', x) for x in content[2:]]
# content = [re.sub('f32', '', x) for x in content]
# content = [np.fromstring( x, dtype=np.float, sep=',' ) for x in content]
# flat_list = np.array([val for row in content for val in row])
# tomo_lib.savebackprojection("output//sirt.png", flat_list, 64)

with open("output//projdifftest") as f:
    content = f.readlines()
print(content.shape)
content =  content[0].strip()
print(content.shape)
content = re.sub('\[', '',  content[0])
print(content.shape)
content = re.sub('\]', '',  content)
print(content.shape)
content = re.sub('f32', '',  content)
print(content.shape)
content = np.array([np.fromstring( content, dtype=np.float, sep=',' )])
print(content.shape)
tomo_lib.savesinogram("output//projdifftest.png", content, 512, 30)
