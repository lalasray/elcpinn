# elcpinn
Here’s the general modified equation for the Rogowski coil’s output voltage, incorporating **frequency** and **temperature**:

### General Equation:
\[
u_s(t, f, T) = -M(T) \cdot \frac{dI_p(t)}{dt}
\]
Where:

- \( u_s(t, f, T) \) is the voltage output of the Rogowski coil as a function of time \( t \), frequency \( f \), and temperature \( T \).
- \( M(T) \) is the mutual inductance of the coil, adjusted for temperature:
  \[
  M(T) = M_0 \cdot \left(1 + \alpha(T - T_0)\right)
  \]
  - \( M_0 \) is the nominal mutual inductance at the reference temperature \( T_0 \) (in Henrys per meter or other appropriate units).
  - \( \alpha \) is the temperature coefficient of the coil (typically a small value like \( 0.01 \)).
  - \( T \) is the current temperature in °C.
  - \( T_0 \) is the reference temperature (usually 20°C).
  
- \( \frac{dI_p(t)}{dt} \) is the time derivative of the primary current, which may be influenced by the **frequency** of the signal. For a sinusoidal current input \( I_p(t) = I_0 \sin(\omega t) \), where \( \omega = 2 \pi f \), this derivative becomes:
  \[
  \frac{dI_p(t)}{dt} = I_0 \cdot \omega \cdot \cos(\omega t)
  \]
  Where:
  - \( I_0 \) is the amplitude of the primary current.
  - \( \omega = 2 \pi f \) is the angular frequency of the input current (with \( f \) in Hz).

### Complete Equation with All Unknowns:
\[
u_s(t, f, T) = -M_0 \cdot \left(1 + \alpha(T - T_0)\right) \cdot I_0 \cdot \omega \cdot \cos(\omega t)
\]
Where:
- \( u_s(t, f, T) \): Rogowski coil output voltage as a function of time \( t \), frequency \( f \), and temperature \( T \).
- \( M_0 \): Mutual inductance at reference temperature \( T_0 \).
- \( \alpha \): Temperature coefficient.
- \( T \): Current temperature in °C.
- \( T_0 \): Reference temperature in °C (typically 20°C).
- \( I_0 \): Amplitude of the primary current.
- \( \omega = 2 \pi f \): Angular frequency, where \( f \) is the frequency in Hz.
- \( \cos(\omega t) \): Represents the time behavior of the sinusoidal current.

### Interpretation:
- The **frequency** \( f \) directly affects the output voltage because \( \omega = 2\pi f \) multiplies the derivative of the current.
- The **temperature** \( T \) modifies the mutual inductance \( M(T) \), which in turn affects the output voltage, scaling it up or down depending on the temperature deviation from \( T_0 \).

This is the full equation incorporating both the **frequency** of the input signal and **temperature effects** on the coil’s properties.

Would you like to explore further variations or test this out in code?