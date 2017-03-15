#Blatt4


import numpy as np
import matplotlib.pyplot as plt

def dealiasing(x):
    k= len(x) // 3
    x[k+1:-k]= 0.0
    return x

def sech(x):
    return 1.0/np.cosh(x)

    
t=0    
L = 4.0 * np.pi     
N = 256    
c1 = 5
c2 = 4   
dx = L/N
Tend = 4000           
h = 0.0001                       
x = (np.arange(N)-N/2) * dx 
k = np.zeros(N) 
k[0:N/2+1] = np.arange(N/2+1)*2*np.pi/L
k[N/2+1:] = np.arange(-N/2+1,0)*2*np.pi/L


u0=0.5*c1**2*sech(0.5*c1*(x+2))**2 + 0.5*c2**2*sech(0.5*c2*(x+0.5))**2   
uk = np.fft.fft(u0) 
Uk = np.exp(1j*k**3*t)*uk
    

def rhs(t,Uk): 
    return -3*1j*k*np.exp(-1j*k**3*t)*np.fft.fft( np.fft.ifft(np.exp(1j*k**3*t)*dealiasing(Uk))**2 )
    #return -3*1j*k*np.exp(-1j*k**3*t)*np.fft.rfft( np.fft.irfft(np.exp(1j*k**3*t)*uk)**2 )
t=0
plt.ion()         
# Runge Kutta wie beim Blatt2
for n in range(Tend+1):
    k1 =h*rhs(t,Uk)
    k2 =h*rhs(t+0.5*h, Uk+0.5*k1)
    k3 =h*rhs(t+0.5*h, Uk+0.5*k2)
    k4 =h*rhs(t+h, Uk+k3)
    Uk = Uk + (1.0/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)
    t=t+h
    
    if n % 100 == 0: 
        u = np.fft.ifft(np.exp(1j*k**3*t)*Uk)
        plt.cla() 
        plt.plot(x,u0,'r',linewidth=2.0)
        plt.plot(x,u.real,'b',linewidth=2.0)
        plt.ylabel("u(x,t)")
        plt.xlabel("x")
        plt.title('u(x,'+ str(n) + ')')
        #plt.axis([-L/2, L/2, -1, 1])
        plt.show() 
        #plt.savefig('{}.jpg'.format(n))   #wurde nur zum erstellen der Bilder für die gif gebraucht
        plt.pause(0.1)


plt.ioff()
plt.show()    