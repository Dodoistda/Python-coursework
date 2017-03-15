#Brüsselator
import numpy as np
import matplotlib.pyplot as plt


Lx =200
Ly =200
Nx =128
Ny =128
x, y = np.meshgrid(np.arange(Nx) * Lx/Nx,np.arange(Ny) * Ly/Ny)
kx, ky = np.meshgrid(np.fft.fftfreq(Nx,Lx/(Nx*2.0*np.pi)), np.fft.fftfreq(Ny,Ly/(Ny*2.0*np.pi)))
ksq = kx*kx + ky*ky

alpha=1.0
a=3.0
b=9.9
du1=8.33
du2=46.0
dv1=8.33
dv2=120.0
u1hom=3.0
u2hom=3.0
v1hom=3.3
v2hom=3.3

u1=u1hom+(np.random.random((Nx,Ny))-0.5)*0.1
v1=v1hom+(np.random.random((Nx,Ny))-0.5)*0.1
u2=u2hom+(np.random.random((Nx,Ny))-0.5)*0.1
v2=v2hom+(np.random.random((Nx,Ny))-0.5)*0.1
uF1=np.fft.fft2(u1)
vF1=np.fft.fft2(v1)
uF2=np.fft.fft2(u2)
vF2=np.fft.fft2(v2)


tau=0.01 #Zeitschritt
t_Ende=250
Ende= 25000 #Anzahl Schritte für die inrange Funktion
bild= 500 #eine Bilddatei alle 500 Schritte

#Vorfaktor
vorfaktorU1=1.0/(1.0+tau*du1*ksq)
vorfaktorU2=1.0/(1.0+tau*du2*ksq)
vorfaktorV1=1.0/(1.0+tau*dv1*ksq)
vorfaktorV2=1.0/(1.0+tau*dv2*ksq)




for n in range(Ende):
	uF1= vorfaktorU1*(uF1+tau*np.fft.fft2(np.fft.ifft2(uF1)**2*np.fft.ifft2(vF1)-(b+1.0)*np.fft.ifft2(uF1)+a+alpha*(u2-u1)))
	uF2= vorfaktorU2*(uF2+tau*np.fft.fft2(np.fft.ifft2(uF2)**2*np.fft.ifft2(vF2)-(b+1.0)*np.fft.ifft2(uF2)+a+alpha*(u1-u2)))
	vF1= vorfaktorV1*(vF1+tau*np.fft.fft2(-np.fft.ifft2(uF1)**2*np.fft.ifft2(vF1)+b*np.fft.ifft2(uF1)+alpha*(v2-v1)))
	vF2= vorfaktorV2*(vF2+tau*np.fft.fft2(-np.fft.ifft2(uF2)**2*np.fft.ifft2(vF2)+b*np.fft.ifft2(uF2)+alpha*(v1-v2)))

	u1= np.fft.ifft2(uF1)
	v1= np.fft.ifft2(vF1)
	u2= np.fft.ifft2(uF2)
	v2= np.fft.ifft2(vF2)

	if(n % bild == 0):
		print("Step %04d" %(n/bild))
		plt.suptitle('L='+str(int(Lx))+', N='+str(Nx)+', a'+str(a)+', b'+str(b))
		plt.subplot(2,2,1)
		plt.cla()
		plt.imshow(u1.real)
		plt.ylabel("y",fontsize=18)
		plt.xlabel("x",fontsize=18)
		plt.title('u_1(x, y, '+ str(n) + ')')
		plt.axis('off')
		plt.subplot(2,2,2)
		plt.cla()
		plt.imshow(v1.real)
		plt.title('v_1(x,y,'+str(n)+')')
		plt.axis('off')
		plt.subplot(2,2,3)
		plt.cla()
		plt.imshow(u2.real)
		plt.title('u_2(x,y,'+str(n)+')')
		plt.axis('off')
		plt.subplot(2,2,4)
		plt.cla()
		plt.imshow(v2.real)
		plt.title('v_2(x,y,'+str(n)+')')
		plt.axis('off')
		filename = "SH-2D%04d.jpg" % (n/bild)
		plt.savefig(filename)
		plt.close()