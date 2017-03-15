#Blatt5 1D
#Das Programm gibt nach dem Durchlaufen die fertige Bilddatei aus.

import numpy as np
import matplotlib.pyplot as plt

L = 200.0 
N = 512.0   
T=2000   
h=0.05 


alpha=0.0
beta=1.5



x = (np.arange(N)-N/2) * L/N 
k = np.zeros(N) 
k[0:N/2+1] = np.arange(N/2+1)*2*np.pi/L
k[N/2+1:] = np.arange(-N/2+1,0)*2*np.pi/L


Adata=np.arange((T+1)*N,dtype=np.complex).reshape(T+1,N)
Tdata=np.arange(T+1)  

def sech(x):
    return 1.0/np.cosh(x)

#A0 für die verschiedenen Startbedingungen
A0= 0.0*np.ones(N) + 0.1*(np.random.randn(N)+1j*np.random.randn(N))
#A0= np.sqrt(1-(20*np.pi/L)**2)*np.exp(20*np.pi*x*1j/L)+ 0.1*(np.random.randn(N)+1j*np.random.randn(N))
#A0= 1.0*np.ones(N) + 0.1*(np.random.randn(N))
#A0=sech((x+10.0)**2)+0.8*sech((x-30.00)**2)+ 0.01*(np.random.randn(N))


A=A0
Adata[0,:]=A0
Adach=np.fft.fft(A)

q=1-k**2*(1+1j*alpha)
coef1=((1+q*h)*np.exp(q*h)-1-2*h*q)/(h*q**2)
coef2=(-np.exp(q*h)+1+h*q)/(h*q**2)

for n in range(T):
    Nn=-(1+1j*beta)*np.fft.fft(np.fft.ifft(Adach)*np.abs(np.fft.ifft(Adach))**2)
    if n==0:
        Nn1=Nn
    Adach=Adach*np.exp(q*h)+Nn*coef1+Nn1*coef2
    Nn1=Nn
    A=np.fft.ifft(Adach)
    Adata[n+1,:]=A

        
        
fig1=plt.figure(1, figsize=(16, 12))
fig1.suptitle('$\\alpha$='+str(alpha)+','+'$\\,\\beta=$'+str(beta)+','+' L='+str(L)+', T='+str(T), fontsize=20)
#
plt.subplot(2,2,1)
plt.contourf(x,Tdata,Adata.real,20,cmap=plt.cm.jet,vmin=0.0,vmax=1.0)
plt.ylabel("time",fontsize=20)
plt.xlabel("space",fontsize=20)
plt.title('Re(A)',fontsize=20)
plt.xticks([-L/2, 0, L/2],['-L/2', '0', 'L/2'],fontsize=18)
plt.yticks([0, T/2, T],['0', 'T/2', 'T'],fontsize=18)


plt.subplot(2,2,2)
plt.contourf(x,Tdata,Adata.imag,20,cmap=plt.cm.jet,vmin=0.0,vmax=1.0)
plt.title('|A|',fontsize=20)
plt.ylabel("time",fontsize=20)
plt.xlabel("space",fontsize=20)
plt.xticks([-L/2, 0, L/2],['-L/2', '0', 'L/2'],fontsize=18)
plt.yticks([0, T/2, T],['0', 'T/2', 'T'],fontsize=18)


plt.subplot(2,2,3)
plt.contourf(x,Tdata,abs(Adata),100,cmap=plt.cm.jet,vmin=0.0,vmax=1.0)
plt.title('|A|',fontsize=20)
plt.ylabel("time",fontsize=20)
plt.xlabel("space",fontsize=20)
plt.xticks([-L/2, 0, L/2],['-L/2', '0', 'L/2'],fontsize=18)
plt.yticks([0, T/2, T],['0', 'T/2', 'T'],fontsize=18)


plt.subplot(2,2,4)
plt.contourf(x,Tdata,np.angle(Adata),100,cmap=plt.cm.jet,vmin=0.0,vmax=1.0)
plt.title('arg(A)',fontsize=20)
plt.ylabel("time",fontsize=20)
plt.xlabel("space",fontsize=20)
plt.xticks([-L/2, 0, L/2],['-L/2', '0', 'L/2'],fontsize=18)
plt.yticks([0, T/2, T],['0', 'T/2', 'T'],fontsize=18)
        

filename = "SpaceTime_"+str(alpha)+"_"+str(beta)+".png"
plt.savefig(filename)
plt.close()