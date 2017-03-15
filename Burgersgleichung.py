#Je kleiner nu, desto schneller gerät die Lösung ausser Kontrolle. Ohne dealiasing kann man nu bis
#auf 0.008 stellen, bei 0.007 bricht die Berechnung ab.
#Für festes nu=0.005 verbessert dealiasing in allen Fällen das Ergebnis, bei N=128 und 64 gibt es
#starke Schwankungen, aber zumindest bricht die Berechnung in keinem der Fällt ab wie bei fehlendem dealiasing.

import numpy as np
import matplotlib.pyplot as plt

def dealiasing(x):
    k= len(x) // 3
    x[k+1:-k]= 0.0
    return x
    
    
L = 4.0 * np.pi     
N = 64       
dx = L/N
Tend=4000           
h=0.01             
nu=0.005             
dtdiff = dx*dx/2.0/nu
x = (np.arange(N)-N/2) * dx 
k = np.zeros(N) 
k[0:N/2+1] = np.arange(N/2+1)*2*np.pi/L
k[N/2+1:] = np.arange(-N/2+1,0)*2*np.pi/L


u0=np.sin(x)
uk = np.fft.fft(u0) 
    

def rhs(nu,k,uk): 
    return -nu * k**2 * uk + np.fft.fft(np.fft.ifft(-1j*k*dealiasing(uk))*np.fft.ifft(dealiasing(uk)))
    #return -nu * k**2 * uk + np.fft.fft(np.fft.ifft(-1j*k*uk)*np.fft.ifft(uk))

plt.ion()         
# Runge Kutta wie beim Blatt2
for n in range(Tend+1):
    k1 =rhs(nu,k,uk)
    k2 =rhs(nu,k,uk+0.5*h*k1)
    k3 =rhs(nu,k,uk+0.5*h*k2)
    k4 =rhs(nu,k,uk+h*k3)
    uk = uk + (h/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)
    
    if n % 100 == 0: 
        u = np.fft.ifft(uk)
        plt.cla() 
        plt.plot(x,u0,'r',linewidth=2.0)
        plt.plot(x,u.real,'b',linewidth=2.0)
        plt.ylabel("u(x,t)")
        plt.xlabel("x")
        plt.title('u(x,'+ str(n) + ')')
        plt.axis([-L/2, L/2, -1, 1])
        plt.show() 
        plt.savefig('{}.jpg'.format(n))   #wurde nur zum erstellen der Bilder für die gif gebraucht
        plt.pause(0.1)


plt.ioff()
plt.show()    