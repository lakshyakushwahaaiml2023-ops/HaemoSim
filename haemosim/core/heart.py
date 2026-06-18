import numpy as np

class HeartModel:
    """
    A time-varying elastance left-heart model coupled with a 3-element Windkessel afterload.
    
    The cardiac cycle is represented by a time-varying elastance function E(t) using
    the double-Hill activation curve. Left ventricular pressure and volume are computed
    at each timestep.
    """
    def __init__(self, heart_rate=75.0, E_max=1.5, E_min=0.06, T_systole=0.3,
                 C_arterial=1.2, R_peripheral=1.0, Z_characteristic=0.06,
                 P_atrial=10.0, R_mitral=0.01, V_0=0.0):
        """
        Initialize the HeartModel.
        
        Parameters:
        -----------
        heart_rate : float
            Heart rate in beats per minute (bpm). Default: 75.0.
        E_max : float
            Maximum (end-systolic) elastance in mmHg/mL. Default: 1.5.
        E_min : float
            Minimum (diastolic) elastance in mmHg/mL. Default: 0.06.
        T_systole : float
            Duration of the systolic phase in seconds. Default: 0.3.
        C_arterial : float
            Arterial compliance in mL/mmHg. Default: 1.2.
        R_peripheral : float
            Peripheral resistance in mmHg*s/mL. Default: 1.0.
        Z_characteristic : float
            Characteristic aortic impedance in mmHg*s/mL. Default: 0.06.
        P_atrial : float
            Mean left atrial pressure in mmHg (for filling phase). Default: 10.0.
        R_mitral : float
            Mitral valve resistance in mmHg*s/mL. Default: 0.01.
        V_0 : float
            Unloaded ventricular volume (dead volume) in mL. Default: 0.0.
        """
        self.heart_rate = heart_rate
        self.E_max = E_max
        self.E_min = E_min
        self.T_systole = T_systole
        self.C_arterial = C_arterial
        self.R_peripheral = R_peripheral
        self.Z_characteristic = Z_characteristic
        self.P_atrial = P_atrial
        self.R_mitral = R_mitral
        self.V_0 = V_0
        
        # Cardiac period in seconds
        self.T_period = 60.0 / self.heart_rate
        
        # Shape factors for the double-Hill activation function
        self.n1 = 1.9
        self.n2 = 21.9
        
        # Peak of the unscaled function g(y) where y = t/T_systole occurs near 1.0.
        # With the shape factors 1.9 and 21.9, the peak is approximately 0.64262.
        # The scaling factor k = 1 / max_val normalizes the peak to 1.0.
        self.k = 1.55612792036889
        
        # Initialize simulation state variables
        self.reset()

    def reset(self, V_lv=120.0, P_c=80.0, t=0.0):
        """
        Reset the simulation state variables.
        """
        self.V_lv = V_lv
        self.P_c = P_c
        self.t = t

    def get_elastance(self, t):
        """
        Compute the time-varying elastance E(t) at a given time t.
        """
        t_c = t % self.T_period
        y = t_c / self.T_systole
        
        # Handle boundary case at t=0
        if y == 0.0:
            e_t = 0.0
        else:
            term1 = (y / 0.7) ** self.n1
            term2 = (y / 1.17) ** self.n2
            e_t = self.k * (term1 / (1.0 + term1)) * (1.0 / (1.0 + term2))
            
        return self.E_min + (self.E_max - self.E_min) * e_t

    def step(self, dt, q_in):
        """
        Advance the model state by one timestep dt.
        
        Parameters:
        -----------
        dt : float
            Timestep size in seconds.
        q_in : float
            Inflow rate from the left atrium in mL/s.
            
        Returns:
        --------
        P_aortic : float
            Aortic pressure at the end of the timestep (mmHg).
        V_lv : float
            Left ventricular volume at the end of the timestep (mL).
        Q_aortic : float
            Aortic outflow rate at the end of the timestep (mL/s).
        """
        # Sub-stepping is used to maintain stability in the presence of stiff dynamics
        # (e.g. low characteristic impedance) and piecewise valve transitions.
        sub_steps = 10
        dt_sub = dt / sub_steps
        
        for _ in range(sub_steps):
            E = self.get_elastance(self.t)
            P_lv = E * (self.V_lv - self.V_0)
            
            # Aortic valve outflow (diode)
            q_out = max(0.0, (P_lv - self.P_c) / self.Z_characteristic)
            
            # Update LV Volume
            self.V_lv += (q_in - q_out) * dt_sub
            self.V_lv = max(self.V_0, self.V_lv)
            
            # Update Windkessel compliance pressure P_c
            self.P_c += ((q_out - self.P_c / self.R_peripheral) / self.C_arterial) * dt_sub
            
            self.t += dt_sub
            
        # Compute outputs at the end of the timestep
        E = self.get_elastance(self.t)
        P_lv = E * (self.V_lv - self.V_0)
        Q_aortic = max(0.0, (P_lv - self.P_c) / self.Z_characteristic)
        P_aortic = self.P_c + self.Z_characteristic * Q_aortic
        
        return P_aortic, self.V_lv, Q_aortic

    def simulate(self, duration, dt):
        """
        Simulate the coupled system for a given duration.
        
        Parameters:
        -----------
        duration : float
            Total simulation time in seconds.
        dt : float
            Timestep size in seconds.
            
        Returns:
        --------
        results : dict of np.ndarray
            Time-series arrays of model variables.
        """
        self.reset()
        num_steps = int(np.floor(duration / dt))
        
        # Pre-allocate arrays
        times = np.zeros(num_steps)
        P_aortic = np.zeros(num_steps)
        V_lv = np.zeros(num_steps)
        Q_aortic = np.zeros(num_steps)
        P_lv = np.zeros(num_steps)
        elastance = np.zeros(num_steps)
        
        for i in range(num_steps):
            t_curr = self.t
            times[i] = t_curr
            
            # Compute current elastance and pressure
            E = self.get_elastance(t_curr)
            p_lv_val = E * (self.V_lv - self.V_0)
            
            elastance[i] = E
            P_lv[i] = p_lv_val
            
            # Passive mitral valve filling (diode)
            q_in = max(0.0, (self.P_atrial - p_lv_val) / self.R_mitral)
            
            # Advance state
            p_ao, v_lv, q_ao = self.step(dt, q_in)
            
            # Record step outputs
            P_aortic[i] = p_ao
            V_lv[i] = v_lv
            Q_aortic[i] = q_ao
            
        return {
            "time": times,
            "P_aortic": P_aortic,
            "V_lv": V_lv,
            "Q_aortic": Q_aortic,
            "P_lv": P_lv,
            "elastance": elastance
        }
