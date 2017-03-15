#Brüsselator
import numpy as np
import matplotlib.pyplot as plt


Lx =120 
Ly =120
Nx =128
Ny =128
x, y = np.meshgrid(np.arange(Nx) * Lx/Nx,np.arange(Ny) * Ly/Ny)
kx, ky = np.meshgrid(np.fft.fftfreq(Nx,Lx/(Nx*2.0*np.pi)), np.fft.fftfreq(Ny,Ly/(Ny*2.0*np.pi)))
ksq = kx*kx + ky*ky

a=3.0
b=18.0
du=5.0
dv=12.0
uhom=a
vhom=b/a

u=uhom+(np.random.random((Nx,Ny))-0.5)*0.1
v=vhom+(np.random.random((Nx,Ny))-0.5)*0.1
uF=np.fft.fft2(u)
vF=np.fft.fft2(v)



tau=0.01 #Zeitschritt
t_Ende=250
Ende= 25000 #Anzahl Schritte für die inrange Funktion
bild= 500 #eine Bilddatei alle 500 Schritte

#Vorfaktor
vorfaktorV=1.0/(1.0+tau*du*ksq)
vorfaktorU=1.0/(1.0+tau*du*ksq+tau*(b+1))

for n in range(Ende):
	uF= vorfaktorU*(uF+tau*np.fft.fft2(np.fft.ifft2(uF)**2*np.fft.ifft2(vF)-(b+1.0)*np.fft.ifft2(uF)+a))
	vF= vorfaktorV*(vF+tau*np.fft.fft2(-np.fft.ifft2(uF)**2*np.fft.ifft2(vF)+b*np.fft.ifft2(uF)))
	u= np.fft.ifft2(uF)
	v= np.fft.ifft2(vF)

	if(n % bild == 0):
		print("Step %04d" %(n/bild))
		plt.suptitle('L='+str(int(Lx))+', N='+str(Nx)+', a'+str(a)+', b'+str(b))
		plt.subplot(1,2,1)
		plt.cla()
		plt.imshow(u.real)
		plt.ylabel("y",fontsize=18)
		plt.xlabel("x",fontsize=18)
		plt.title('u(x, y, '+ str(n) + ')')
		plt.axis('off')
		plt.subplot(1,2,2)
		plt.cla()
		plt.imshow(v.real)
		plt.title('v(x,y,'+str(n)+')')
		plt.axis('off')
		filename = "SH-2D%04d.jpg" % (n/bild)
		plt.savefig(filename)
		plt.close()