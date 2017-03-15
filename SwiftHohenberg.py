import numpy as np
import matplotlib.pyplot as plt


Lx = 64 
Ly =64
Nx = 128
Ny = 128
x, y = np.meshgrid(np.arange(Nx) * Lx/Nx,np.arange(Ny) * Ly/Ny)
kx, ky = np.meshgrid(np.fft.fftfreq(Nx,Lx/(Nx*2.0*np.pi)), np.fft.fftfreq(Ny,Ly/(Ny*2.0*np.pi)))
ksq = kx*kx + ky*ky

#Startbedingungen für Funktion und epsilon/delta
#psi = 0.0+(np.random.random((Nx,Ny))-0.5)*0.1 #Rauschen für Aufgabe 1
#psi=np.sin(0.88357*x)+(np.random.random((Nx,Ny))-0.5)*0.1 #Aufgabe2 Zigzag
psi=np.sin(1.178*x)+(np.random.random((Nx,Ny))-0.5)*0.1   #Aufgabe2 Eckhaus 


psiF= np.fft.fft2(psi)

#Streifen 0.3 und 0; Hexagone 0.1 und 1; Zigzag/Eckhaus 0.3 und 0
epsilon=0.3
delta=0.0

tau=0.01 #Zeitschritt
t_Ende=500
Ende= 50000 #Anzahl Schritte für die inrange Funktion
bild= 1000 #eine Bilddatei alle 500 Schritte

def L(ksq):
	lam=epsilon-(1.0-ksq)**2
	return lam

Vorfaktor= 1.0/((1.0/tau)-L(ksq))


for n in range(Ende):
	psiF= Vorfaktor*((psiF/tau)+(delta*np.fft.fft2(np.fft.ifft2(psiF)**2)-(np.fft.fft2(np.fft.ifft2(psiF)**3))))
	psi= np.fft.ifft2(psiF)

	if(n % bild == 0):
		print("Step %04d" %(n/bild))
		plt.suptitle('L='+str(int(Lx))+', N='+str(Nx)+', $\\varepsilon=$'+str(epsilon)+', $\\delta=$'+str(delta))
		plt.subplot(1,2,1)
		plt.cla()
		plt.imshow(psi.real)
		plt.ylabel("y",fontsize=18)
		plt.xlabel("x",fontsize=18)
		plt.title('$\\psi$(x, y, '+ str(n) + ')')
		plt.axis('off')
		plt.subplot(1,2,2)
		plt.cla()
		plt.imshow(np.abs(np.fft.fftshift(psiF)))
		plt.title('Fourier space, $|\\widehat{\\psi}_k|$')
		plt.axis('off')
		filename = "SH-2D%04d.jpg" % (n/bild)
		plt.savefig(filename)
		plt.close()