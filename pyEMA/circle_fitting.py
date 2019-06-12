import numpy as np

class CircleFitting:

    def __init__(self, frf, n_avg, n_mid):
        self.frf = frf
        self.n_avg = n_avg
        self.n_mid = n_mid


    def fit_the_circle(self, alpha):
        '''
        Izračuna parametre podanega kroga po metodi najmanjših kvadratnih odstopanj.
        
        Args:
            alpha (complex array): Receptanca sistema.
        
        Returns:
            (x0, y0) (float, tuple): Koordinati središča kroga.
            R (float): Polmer kroga.
        '''
        A = np.zeros((3,3))
        d = np.zeros(3)

        L =  len(alpha)
        x = np.real(alpha)
        y = np.imag(alpha)
        
        A[0,0] = np.sum(x**2)
        A[0,1] = np.sum(x*y)
        A[0,2] = -np.sum(x)
        A[1,0] = A[0,1]
        A[1,1] = np.sum(y**2)
        A[1,2] = -np.sum(y)
        A[2,0] = A[0,2]
        A[2,1] = A[1,2]
        A[2,2] = L
        
        d[0] = -np.sum(x * (x**2 + y**2))
        d[1] = -np.sum(y * (x**2 + y**2))
        d[2] = np.sum(x**2 + y**2)
        
        abc = np.linalg.solve(A,d)
        
        x0 = -abc[0]/2
        y0 = -abc[1]/2
        R = np.sqrt(abc[2] + x0**2 + y0**2)
        
        return x0+y0*1.j, R


    def get_natural_frequency(self, alpha, c):
        a = alpha[1:]
        b = np.roll(a, 1)
        
        v1 = np.array([np.real(a)-np.real(c), np.imag(a)-np.imag(c)])
        v2 = np.array([np.real(b)-np.real(c), np.imag(b)-np.imag(c)])
        v3 = np.array([np.real(a)-np.real(b), np.imag(a)-np.imag(b)])
        
        lv1 = np.sqrt(np.sum(v1**2, axis=0))
        lv2 = np.sqrt(np.sum(v2**2, axis=0))
        lv3 = np.sqrt(np.sum(v3**2, axis=0))
        
        gamma = np.arccos((lv1**2 + lv2**2 - lv3**2)/(2*lv1*lv2))
        
        self.nf_idx = np.argmax(gamma)