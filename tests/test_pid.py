import pytest
from project.algorithms.pid_alg import PID

class TestPID(object):

    def test_pid_single_timestep(self):
        pid   = PID(timestep=1/60, consts=(1,1,1))
        error = pid.calculate(6)
        assert error == 6.1

    def test_pid_multiple_timesteps(self):
        pid   = PID(timestep=1/60, consts=(1,1,1))
        
        for i in range(59):
            pid.calculate(6)

        error = pid.calculate(6)
        assert round(error,5) == 12

    def test_pid_integral_error(self):
        pid   = PID(timestep=1/60, consts=(0,1,0))
        pid.calculate(6) 
        error = pid.calculate(6)
        assert error == 0.2

    def test_pid_derivative_error(self): 
        pid   = PID(timestep=1/60, consts=(0,0,1))
        pid.calculate(6) 
        error = pid.calculate(6.01) 
        assert round(error,5) == 0.6
    
    def test_pid_custom_derivative_error(self): 
        pid   = PID(timestep=1/60, consts=(0,0,1))
        pid.calculate(6) 
        error = pid.calculate(6, de=0.6) 
        assert round(error,5) == 0.6

    def test_pid_proportional_error(self): 
        pid   = PID(timestep=1/60, consts=(1,0,0))
        error = pid.calculate(5) 
        assert round(error,5) == 5

    def test_pid_Kp(self): 
        pid   = PID(timestep=1/60, consts=(2,1,1))
        error = pid.calculate(6)
        assert error == 12.1

    def test_pid_Ki(self): 
        pid   = PID(timestep=1/60, consts=(2,2,1))
        error = pid.calculate(6)
        assert error == 12.2

    def test_pid_Kd(self):
        pid   = PID(timestep=1/60, consts=(0,0,2))
        pid.calculate(6) 
        error = pid.calculate(6.01) 
        assert round(error,5) == 1.2




