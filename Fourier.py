#Die Datei sollte alle Graphen nacheinander ausgeben.

import numpy as np
import matplotlib.pyplot as mpl

Unterteilung=2*np.pi/129
Ende=2*np.pi + Unterteilung
a=np.arange(0,Ende,Unterteilung)

f=np.sin(a)


print(f)
plotf=mpl.plot(a,f)  #Ausgabe Originalgraph
mpl.show(plotf)

fourierf=np.fft.fft(f) #Fouriertransformation
plotfourierf=mpl.plot(a,fourierf) #Ausgabe Fouriertransformierte
mpl.show(plotfourierf)


rfourierf=np.fft.ifft(fourierf) #Ruecktransformation
plotrfourierf=mpl.plot(a,rfourierf)
mpl.show(plotrfourierf)





g= np.exp(-(a-np.pi)**2)
plotg=mpl.plot(a,g)
mpl.show(plotg)

fourierg=np.fft.fft(g) #Fouriertransformation
plotfourierg=mpl.plot(a,fourierg) #Ausgabe Fouriertransformierte
mpl.show(plotfourierg)


rfourierg=np.fft.ifft(fourierg) #Ruecktransformation
plotrfourierg=mpl.plot(a,rfourierg)
mpl.show(plotrfourierg)



k=np.fft.fftfreq(130,Unterteilung)
Ableitung=1j*k*fourierg
Ableitung2=np.fft.ifft(Ableitung)
plotableitung=mpl.plot(a,Ableitung2)
mpl.show(plotableitung)

#Aus irgendeinem Grund spuckt mir diese Version einen ordentlichen Graphen für g(x) aus (zumindest laut wolframalpha), aber wenn ich statt g(x) f(x) verwende ist das Ergebnis Blödsinn und ich versteh nicht wieso.