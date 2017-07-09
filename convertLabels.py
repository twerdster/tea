import numpy as np
with open('data/Labels.lbl',mode='rb') as file:
    data=file.read()

a=np.fromstring(data,np.int8)
print np.max(a)

b=np.zeros(a.shape)
b[a<5]=a[a<5]+np.random.randint(4,size=a[a<5].shape)*5
b[a>=5]=20+(a[a>=5]-5)*6+np.random.randint(6,size=a[a>=5].shape)
b=b.astype(np.int8)
data=np.ndarray.tostring(b)
with open('data/Labels.lbl',mode='wb') as file:
    file.write(data)
