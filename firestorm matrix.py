# Simulating Firestorm Matrix stabilization system with control loops and Euler integration






# Import numpy at the top to ensure availability
try:
    import numpy as np
except ImportError as e:
    raise ImportError("Required packages are missing. Please install them using 'pip install numpy'.") from e

# Defer matplotlib import to runtime to avoid IDE import errors
plt = None
def import_matplotlib():
    global plt
    try:
        import matplotlib.pyplot as _plt
        plt = _plt
    except ImportError as e:
        raise ImportError("matplotlib is required for plotting. Please install it using 'pip install matplotlib'.") from e

# Simulation parameters
time_steps = 100
dt = 0.1  # time step size

# System thresholds and constants
phase_error_threshold = 0.01
max_temperature = 350  # Kelvin
initial_phase_error = 0.05
initial_temperature = 300  # Kelvin
initial_thrust = 0.01  # 1%

# Constants for system dynamics
k_phase_correction = 0.05       # PLL pull strength
k_temp_increase = 2.0           # heating effect from core/magnetic activity
k_water_cooling = 0.1           # cooling efficiency per unit flow
k_magnetic_gain = 10.0          # magnetic field response to phase/temperature
k_water_gain = 0.05             # water flow response to temperature rise
k_thrust_gain = 0.02            # thrust ramp rate (gated on stability)



# Import numpy at runtime with error handling
try:
    import numpy as np
except ImportError as e:
    raise ImportError("Required packages are missing. Please install them using 'pip install numpy'.") from e


# Use numpy.typing for better type annotation
from numpy.typing import NDArray
phase_error: NDArray[np.float64] = np.zeros(time_steps)
temperature: NDArray[np.float64] = np.zeros(time_steps)
thrust: NDArray[np.float64] = np.zeros(time_steps)
magnetic_field: NDArray[np.float64] = np.zeros(time_steps)
water_flow: NDArray[np.float64] = np.zeros(time_steps)

# Set initial conditions
phase_error[0] = initial_phase_error
temperature[0] = initial_temperature
thrust[0] = initial_thrust

# Simulation loop
for t in range(1, time_steps):
    # Update magnetic field strength (circulatory dampers)
    magnetic_field[t] = k_magnetic_gain * phase_error[t-1] * (temperature[t-1] / 300)

    # Update water flow rate (moderation jacket)
    water_flow[t] = k_water_gain * (temperature[t-1] - 300)

    # Update temperature (heat from core minus water cooling)
    temp_change = k_temp_increase * magnetic_field[t] - k_water_cooling * water_flow[t]
    temperature[t] = temperature[t-1] + dt * temp_change

    # Update phase error via PLL (exponential pull to zero)
    phase_correction: float = -k_phase_correction * phase_error[t-1]
    phase_error[t] = phase_error[t-1] + dt * phase_correction

    # Gate thrust on stability thresholds
    if abs(float(phase_error[t])) < phase_error_threshold and float(temperature[t]) < max_temperature:
        thrust[t] = thrust[t-1] + k_thrust_gain * dt
    else:
        thrust[t] = thrust[t-1] - k_thrust_gain * dt

    # Clamp thrust between 0 and 1
    thrust_val = float(thrust[t])
    thrust[t] = max(0.0, min(1.0, thrust_val))


if __name__ == "__main__":
    import_matplotlib()
    plt.style.use('seaborn-v0_8')

    from typing import Any
    import numpy as np
    fig: Any
    axs: np.ndarray[Any, Any]
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    axs[0].plot(phase_error, label='Phase Error')
    axs[0].set_ylabel('Phase Error')
    axs[0].legend()

    axs[1].plot(temperature, label='Temperature (K)', color='orange')
    axs[1].set_ylabel('Temperature (K)')
    axs[1].legend()

    axs[2].plot(magnetic_field, label='Magnetic Field Strength', color='green')
    axs[2].set_ylabel('Magnetic Field')
    axs[2].legend()

    axs[3].plot(thrust, label='Thrust Level', color='red')
    axs[3].set_ylabel('Thrust')
    axs[3].set_xlabel('Time Step')
    axs[3].legend()

    plt.tight_layout()
    plt.show()
