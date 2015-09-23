from numpy import *

def MakeHRF(dt, nFrames, 
            peak1=5.4, fwhm1=5.2, 
            peak2=10.8, fwhm2=7.35, 
            dip=0.35):
    t = arange(nFrames)*dt
    
    alpha1 = power(peak1,2)/power(fwhm1,2)*8*log(2)
    beta1 = power(fwhm1,2)/peak1/8/log(2)
    gamma1 = power(t/peak1,alpha1)*exp(-(t-peak1)/beta1)

    alpha2 = power(peak2,2)/power(fwhm2,2)*8*log(2)
    beta2 = power(fwhm2,2)/peak2/8/log(2)
    gamma2 = power(t/peak2,alpha2)*exp(-(t-peak2)/beta2)

    return gamma1 - dip*gamma2

def WienerDeconv(a, h, snr):
    fa = fft.fft(a)
    fh = fft.fft(h)
    fg = conjugate(fh) / (power(abs(fh),2) + 1./snr)
    return fft.ifft(fa*fg)
    

