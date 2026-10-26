"""Independent NumPy WGC99 Taylor kernel/analytic potential for DFTpy.

DFTpy dev fbc47b4e does not provide native WGC. This explicit adapter lets its
optimizer run WGC without calling any GradSCF KEDF code or JAX derivative.
The kernel is integrated directly in eta, independently of GradSCF's log-eta
Hermite table. Equations/conventions: WGC PRB60,16350 and BSD libKEDF.
"""
import math
import numpy as np
from scipy.integrate import solve_ivp


class WGCReference:
    def __init__(self, wavevectors, mesh, volume, reference_density):
        reference_density = np.asarray(reference_density)
        if reference_density.shape != ():
            raise ValueError('WGC reference_density must be a scalar.')
        if not np.isfinite(reference_density) or reference_density <= 0:
            raise ValueError('WGC reference_density must be finite and positive.')
        self.mesh=tuple(mesh);self.dvol=volume/np.prod(mesh)
        self.reference_density=reference_density
        self.alpha=(5+math.sqrt(5))/6;self.beta=(5-math.sqrt(5))/6
        gamma=2.7
        eta=np.linalg.norm(np.asarray(wavevectors),axis=-1)/(2*(3*np.pi**2*reference_density)**(1/3))
        self.kernels=self._kernels(eta,gamma)

    def _kernels(self,eta,gamma):
        c=36*self.alpha*self.beta
        def response(x):
            if x>2:
                z=x**-2
                h=sum(z**k/((2*k+3)*(2*k+5)) for k in range(32))
                return -1-3*h/(1/3+z*h)
            if x<1e-3:return -8/3*x*x+8/45*x**4
            if x==1:return -2.
            return 1/(.5+(1-x*x)/(4*x)*np.log(abs((1+x)/(1-x))))-1-3*x*x
        def rhs(x,y):
            return [y[1],(20*response(x)-(gamma-9)*x*y[1]-c*y[0])/x**2]
        a2=20*(-24/175)/(4-2*(gamma-10)+c)
        a4=20*(-8/125)/(16-4*(gamma-10)+c)
        start=max(200.,float(eta.max())*2)
        y0=[-32/c+a2/start**2+a4/start**4,-2*a2/start**3-4*a4/start**5]
        minimum=max(1e-6,float(eta[eta>0].min())*.5)
        sol=solve_ivp(rhs,(start,minimum),y0,method='DOP853',rtol=2e-11,atol=2e-13,dense_output=True)
        if not sol.success:raise RuntimeError(sol.message)
        safe=np.maximum(eta,minimum)
        w,deta=sol.sol(safe)
        d1=eta*deta
        # eta² w_etaeta from the defining ODE.
        d2=np.array([20*response(x) for x in safe])-(gamma-9)*d1-c*w
        w=np.where(eta>0,w,0.);d1=np.where(eta>0,d1,0.);d2=np.where(eta>0,d2,0.)
        rho=self.reference_density
        return tuple(k.reshape(self.mesh) for k in
            (w,-d1/(6*rho),(d2+(7-gamma)*d1)/(36*rho**2),(d2+(1+gamma)*d1)/(36*rho**2)))

    def evaluate(self,density):
        n=np.maximum(np.asarray(density).reshape(self.mesh),1e-18)
        t=n-self.reference_density
        a=n**self.alpha;b=n**self.beta
        da=self.alpha*n**(self.alpha-1);db=self.beta*n**(self.beta-1)
        def blocks(f):
            f0=np.fft.fftn(f);f1=np.fft.fftn(t*f);f2=np.fft.fftn(.5*t*t*f)
            k0,k1,k2,k12=self.kernels
            return (np.fft.ifftn(k0*f0+k1*f1+k2*f2).real,
                    np.fft.ifftn(k1*f0+k12*f1).real,np.fft.ifftn(k2*f0).real)
        aa=blocks(a);bb=blocks(b)
        energy=np.sum(b*(aa[0]+t*aa[1]+.5*t*t*aa[2]))*self.dvol
        potential=(db*aa[0]+(b+t*db)*aa[1]+(t*b+.5*t*t*db)*aa[2]
                   +da*bb[0]+(a+t*da)*bb[1]+(t*a+.5*t*t*da)*bb[2])
        ctf=.3*(3*np.pi**2)**(2/3)
        return ctf*energy,ctf*potential

    def __call__(self,rho,calcType=('E','V'),**kwargs):
        del calcType,kwargs
        from dftpy.field import DirectField
        from dftpy.functional.functional_output import FunctionalOutput
        energy,potential=self.evaluate(rho)
        return FunctionalOutput(name='WGC99-second-order-nonlocal',energy=energy,
            potential=DirectField(grid=rho.grid,griddata_3d=potential))
