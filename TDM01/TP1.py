# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 10:26:30 2015

@author: zwang04
"""



a = 5;

if a==3:
    b=1
else:
    b=2

print(b)    

for i in [1,3,5]:
    print(i)
    
for i in range(5): # 1 2 3 4
    print(i)        

i=0    
while i<5:
    print(i)
    i+=1   
    
# créer une function
def increment(x):
    print(x)
    return x+1

print(increment(121212121212)) 

# List
a =[1,5,1,8]

print(a[1])   

a[-1]
a[1:3]

a =[1,"sdsdsd",8]

a =[1,5,[1,8]]

# dictionnaire
a={'a':223,'n':222}
a['n']

a={1:'toto','tata':1}

# string
toto = 'toto'+'tata'
l=len(toto)
start = toto.startswith('tqsqsqs')

#

a=array((3,4))
#a=array((1,2,3),(4,5,6),(7,8,9))
a=array(((3,4),(5,6)))

a =zeros((3,)) # une dimension et 3 élements
a =zeros((3,5))
a =zeros((3,5,7))
a.shape

b=randn(3,4,5)

sh1 = b[0].shape
sh2 = b[:,:,0].shape
sh3 = b[:,0,:].shape

b = randn(3,4)
b[0]
b[:,0] # bu zhi dao shi ligne hai shi colonne
b[:,0:1] # garder la dimension

a=randn(3,4)
b=randn(4,5)
c=randn(3,4)
a.dot(b)
d=randn(5,2)
a.dot(b).dot(d)

a.dot(c[0])
c[0]
c[:,0]

x=linspace(-2*pi,2*pi,100)

plot(x,sin(x))



