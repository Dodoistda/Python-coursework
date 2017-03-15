#Blatt52D


import numpy as np
import matplotlib.pyplot as plt


Lx = 100 
Ly =100
Nx = 256
Ny = 256
T_End = 500
tt=50
dt=0.05
#N_t = int(T_End / dt)


alpha=0.0
beta=1.5


x, y = np.meshgrid(np.arange(Nx) * Lx/Nx,np.arange(Ny) * Ly/Ny) 
kx, ky = np.meshgrid(np.fft.fftfreq(Nx,Lx/(Nx*2.0*np.pi)), np.fft.fftfreq(Ny,Ly/(Ny*2.0*np.pi)))
ksq = kx*kx + ky*ky

A0= 1.0 + (np.random.random((Nx,Ny))-0.5)*0.1
A=A0
Adach=np.fft.fft2(A)


q=1-ksq**2*(1+1j*alpha)
coef1=((1+q*dt)*np.exp(q*dt)-1-2*dt*q)/(dt*q**2)
coef2=(-np.exp(q*dt)+1+dt*q)/(dt*q**2)


for n in range(T_End):
    Nn=-(1+1j*beta)*np.fft.fft2(np.fft.ifft2(Adach)*np.abs(np.fft.ifft2(Adach))**2)
    if n==0:
        Nn1=Nn
    Adach=Adach*np.exp(q*dt)+Nn*coef1+Nn1*coef2
    Nn1=Nn
    A=np.fft.ifft2(Adach)
    if (n % tt == 0):
        
        fig1=plt.figure(1, figsize=(16, 12))
        fig1.suptitle('Frozen States:$\\,\\alpha$='+str(alpha)+','+'$\\,\\beta=$'+str(beta), fontsize=20)
        plt.subplot(1,2,1)
        plt.cla()
        plt.imshow(A.real)
        plt.ylabel("y",fontsize=18)
        plt.xlabel("x",fontsize=18)
        plt.title('Re(A(x, y, '+ str(n) + '))')
       

        plt.subplot(1,2,2)
        plt.cla()
        plt.imshow(np.abs(A))
        plt.title('|A|')
        plt.ylabel("y",fontsize=18)
        plt.xlabel("x",fontsize=18)


        filename = "GL-2D%04d.jpg" % (n/tt)
        plt.savefig(filename)
        plt.close()