"""
generate_dataset.py
--------------------
Builds data/knowledge_base.csv — the curated knowledge base that grounds
DMHardware AI Chatbot's Gemini responses.

Every record below is ORIGINAL reference content (written for this project,
not copied from any book/article/datasheet), organized as short,
retrieval-friendly chunks: one clear concept per row.

DATA SOURCES: Original references plus sourced lesson passages and labeled
worked calculations from expand_knowledge.py. Regeneration checks uniqueness.

HOW TO ADD MORE RECORDS:
  1. Add more dicts to any of the *_records lists below, OR
  2. Drop your own notes/datasheets/course material into data/raw_sources/
     (as .txt or .csv) and extend `load_extra_sources()` at the bottom to
     ingest them into the same schema, OR
  3. Point the loader at exported CSVs from your own research (see
     load_extra_sources for the expected columns).

Schema (columns written to knowledge_base.csv):
  id, category, subcategory, title, content, tags
"""

import csv
import os
import json
from collections import Counter
from expand_knowledge import expanded_records

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_PATH = os.path.join(OUT_DIR, "knowledge_base.csv")


# ---------------------------------------------------------------------------
# ELECTRONICS
# ---------------------------------------------------------------------------
electronics_records = [
    ("Fundamentals", "Ohm's Law", "Ohm's Law relates voltage, current, and resistance in a circuit: V = I x R. Given any two values you can solve for the third — for example, a 9V supply across a 470 ohm resistor draws about 19.1 mA."),
    ("Fundamentals", "Power Dissipation", "Electrical power in a resistive element is P = V x I, which can also be written as P = I^2 x R or P = V^2 / R. Exceeding a resistor's rated wattage causes overheating and drift or failure."),
    ("Fundamentals", "Kirchhoff's Current Law (KCL)", "KCL states that the sum of currents entering a node equals the sum of currents leaving it — charge cannot accumulate at a node in steady state. It's the basis for nodal analysis."),
    ("Fundamentals", "Kirchhoff's Voltage Law (KVL)", "KVL states that the sum of voltage rises and drops around any closed loop in a circuit equals zero. It's the basis for mesh/loop analysis."),
    ("Fundamentals", "Series vs Parallel Resistors", "Resistors in series add directly: Rtotal = R1 + R2 + ... In parallel, conductances add: 1/Rtotal = 1/R1 + 1/R2 + ... Series connections divide voltage; parallel connections divide current."),
    ("Fundamentals", "Thevenin Equivalent", "Any linear two-terminal network can be reduced to a single voltage source (Vth) in series with a single resistance (Rth) as seen from those terminals — useful for simplifying load analysis."),
    ("Fundamentals", "Norton Equivalent", "The Norton equivalent represents a linear network as a current source (In) in parallel with a resistance (Rn). Vth = In x Rn relates it directly to the Thevenin equivalent."),
    ("Passive Components", "Resistor Color Code (4-band)", "4-band resistors encode value as: band1-band2 = significant digits, band3 = multiplier (power of ten), band4 = tolerance. Example: brown-black-red-gold = 10 x 100 = 1kOhm +/-5%."),
    ("Passive Components", "Resistor Color Code (5-band)", "5-band precision resistors use three significant digit bands, then a multiplier band, then a tolerance band, giving finer resolution than the 4-band scheme (commonly +/-1% or better)."),
    ("Passive Components", "Capacitor Basics", "A capacitor stores energy in an electric field between two conductive plates separated by a dielectric. Capacitance C = Q/V, measured in farads; typical circuit values range from picofarads to thousands of microfarads."),
    ("Passive Components", "Capacitor Types", "Ceramic capacitors are small, cheap, and good for decoupling/high frequency. Electrolytic capacitors offer large capacitance in small volume but are polarized and have limited lifetime. Film and tantalum capacitors offer stability and low leakage for precision analog work."),
    ("Passive Components", "Decoupling Capacitors", "Small ceramic capacitors (commonly 0.1uF) placed close to an IC's power pins suppress high-frequency noise and supply transients, keeping the local supply rail stable during fast switching."),
    ("Passive Components", "Inductor Basics", "An inductor stores energy in a magnetic field and opposes changes in current. Inductance L relates voltage and current rate of change: V = L x (dI/dt). Common uses include filtering, energy storage in switching regulators, and RF tuning."),
    ("Passive Components", "RC Time Constant", "In a resistor-capacitor circuit, the time constant tau = R x C describes how quickly the capacitor charges or discharges — after one tau the voltage has moved about 63% of the way to its final value."),
    ("Passive Components", "RL Time Constant", "In a resistor-inductor circuit, tau = L / R describes the rate of current rise or decay — after one tau the current has moved about 63% toward its steady-state value."),
    ("Semiconductors", "Diode Basics", "A diode conducts current primarily in one direction. Silicon diodes have roughly a 0.6-0.7V forward voltage drop; germanium diodes are lower (~0.3V); Schottky diodes are lower still (~0.2-0.3V) with faster switching."),
    ("Semiconductors", "Zener Diode", "A Zener diode is designed to operate in reverse breakdown at a well-defined voltage, making it useful for simple voltage regulation or reference circuits when paired with a series current-limiting resistor."),
    ("Semiconductors", "LED Basics", "LEDs emit light when forward biased, with forward voltage depending on color/material (roughly 1.8V for red up to 3.2V+ for blue/white). They require a current-limiting resistor or driver to avoid exceeding rated current."),
    ("Semiconductors", "BJT Overview", "Bipolar Junction Transistors (NPN/PNP) are current-controlled devices where a small base current controls a larger collector-emitter current. Common configurations are common-emitter (gain), common-collector (buffer), and common-base (high-frequency)."),
    ("Semiconductors", "MOSFET Overview", "MOSFETs are voltage-controlled switches/amplifiers. N-channel MOSFETs turn on when gate-source voltage exceeds a threshold; they dominate modern switching power supplies and digital logic due to low gate drive power."),
    ("Semiconductors", "Transistor as a Switch", "When used as a digital switch, a transistor is driven fully into saturation (BJT) or fully enhanced (MOSFET) for the 'on' state, and fully off (cutoff) for the 'off' state, minimizing power dissipation in both states."),
    ("Semiconductors", "Op-Amp Basics", "An ideal operational amplifier has infinite gain, infinite input impedance, and zero output impedance, driving its two inputs to the same voltage via negative feedback (the 'virtual short' concept)."),
    ("Semiconductors", "Op-Amp Inverting Amplifier", "In an inverting amplifier, gain = -Rf/Rin, where Rin connects the source to the inverting input and Rf feeds back from output to inverting input. The non-inverting input is typically grounded or biased."),
    ("Semiconductors", "Op-Amp Non-Inverting Amplifier", "In a non-inverting amplifier, gain = 1 + Rf/Rin. The signal drives the non-inverting input directly, giving high input impedance and a positive (non-inverted) gain."),
    ("Circuits", "Voltage Divider", "A voltage divider made of two series resistors produces Vout = Vin x (R2 / (R1 + R2)). Loading effects from whatever connects to the output can significantly change this ratio if the load impedance is comparable to R2."),
    ("Circuits", "Low-Pass Filter (RC)", "A simple RC low-pass filter passes frequencies below its cutoff f_c = 1/(2*pi*R*C) and attenuates higher frequencies at roughly 20 dB/decade beyond that point."),
    ("Circuits", "High-Pass Filter (RC)", "A simple RC high-pass filter blocks frequencies below cutoff f_c = 1/(2*pi*R*C) and passes higher frequencies, useful for removing DC offset or low-frequency noise from a signal."),
    ("Circuits", "Linear Voltage Regulator", "Linear regulators (e.g., 78xx series) drop excess voltage as heat to produce a stable output, offering low noise and simplicity but poor efficiency when the input-output differential is large."),
    ("Circuits", "Switching Voltage Regulator", "Switching (buck/boost) regulators use an inductor/capacitor and a switching element to convert voltage with high efficiency (often >85%), at the cost of more complex design and higher output ripple/noise than linear regulators."),
    ("Circuits", "Buck Converter", "A buck converter steps voltage down by rapidly switching an input source across an inductor and filtering with a capacitor, with output roughly Vout = Vin x duty_cycle for continuous conduction mode."),
    ("Circuits", "Boost Converter", "A boost converter steps voltage up by storing energy in an inductor during the switch-on phase and releasing it in series with the input during the switch-off phase, giving Vout > Vin."),
    ("Measurement", "Multimeter Basics", "A digital multimeter measures voltage (parallel across a component), current (in series with the circuit), and resistance (with the circuit unpowered), and is the primary hand tool for verifying circuit behavior."),
    ("Measurement", "Oscilloscope Basics", "An oscilloscope displays voltage over time, letting you observe waveform shape, frequency, rise/fall time, noise, and timing relationships between signals — essential for debugging anything with fast-changing signals."),
    ("Sensors", "Analog vs Digital Sensors", "Analog sensors (e.g., a thermistor or potentiometer) output a continuous voltage/current proportional to a measured quantity and typically need an ADC; digital sensors output discrete values or communicate over a protocol like I2C/SPI/UART."),
    ("Sensors", "I2C Protocol", "I2C is a two-wire (SDA/SCL) synchronous serial bus supporting multiple devices on the same lines via unique addresses, commonly used to connect low-speed sensors and peripherals to a microcontroller."),
    ("Sensors", "SPI Protocol", "SPI is a synchronous serial protocol using separate clock, data-in, data-out, and chip-select lines, offering higher throughput than I2C at the cost of more wires and typically point-to-point wiring per device."),
    ("Sensors", "UART Protocol", "UART is an asynchronous serial protocol using separate TX/RX lines with an agreed baud rate rather than a shared clock, commonly used for simple point-to-point communication like a microcontroller talking to a PC or GPS module."),
]

# ---------------------------------------------------------------------------
# ROBOTICS
# ---------------------------------------------------------------------------
robotics_records = [
    ("Actuators", "DC Motor Basics", "A brushed DC motor converts electrical energy to rotational motion via a magnetic field and commutator; speed is roughly proportional to applied voltage and torque relates to current, making them simple to control with a single H-bridge."),
    ("Actuators", "Brushless DC Motor (BLDC)", "BLDC motors move the commutation electronics outside the motor, using an electronic speed controller to sequence current through stator coils. They offer higher efficiency and longevity than brushed motors but need more complex control."),
    ("Actuators", "Stepper Motor Basics", "Stepper motors move in discrete angular steps by energizing coil pairs in sequence, allowing open-loop position control without an encoder, at the cost of lower top speed and possible step loss under excessive load."),
    ("Actuators", "Servo Motor Basics", "An RC-style servo combines a small DC motor, gearbox, and feedback potentiometer with onboard control electronics that accept a PWM position command, making it convenient for simple positional control in robotics."),
    ("Actuators", "H-Bridge Motor Driver", "An H-bridge uses four switching elements arranged so current can flow through a motor in either direction, enabling forward/reverse control and, combined with PWM, variable speed control of a DC motor."),
    ("Actuators", "PWM for Motor Control", "Pulse-width modulation varies the average power delivered to a motor by rapidly switching full voltage on and off at a fixed frequency, with duty cycle controlling effective speed or position command."),
    ("Kinematics", "Forward Kinematics", "Forward kinematics computes the position and orientation of a robot's end effector given known joint angles/lengths, typically via a chain of transformation matrices (e.g., Denavit-Hartenberg parameters)."),
    ("Kinematics", "Inverse Kinematics", "Inverse kinematics computes the joint angles needed to reach a desired end-effector position and orientation — often having multiple valid solutions (elbow-up/elbow-down) and sometimes no closed-form solution, requiring numerical methods."),
    ("Kinematics", "Degrees of Freedom (DOF)", "Degrees of freedom describe the number of independent ways a robot or joint can move. A typical robotic arm for full 3D position and orientation control needs at least 6 DOF."),
    ("Kinematics", "Denavit-Hartenberg Parameters", "DH parameters (link length, link twist, link offset, joint angle) provide a standardized way to describe the geometry of a serial robotic arm's links and joints for kinematic modeling."),
    ("Control", "PID Controller", "A PID controller computes a control output from Proportional (current error), Integral (accumulated past error), and Derivative (rate of error change) terms, widely used for motor speed/position and other feedback control loops."),
    ("Control", "Tuning a PID Loop", "Common tuning approaches include starting with proportional gain only, then adding integral gain to remove steady-state error, then adding derivative gain to reduce overshoot/oscillation. Ziegler-Nichols is a classic systematic tuning method."),
    ("Control", "Open-Loop vs Closed-Loop Control", "Open-loop control applies a command without checking the actual result (e.g., basic stepper motion), while closed-loop control uses sensor feedback to correct errors in real time (e.g., encoder-based position control)."),
    ("Control", "Feedback and Encoders", "Rotary encoders (optical or magnetic) report shaft position or speed, enabling closed-loop control. Incremental encoders report relative motion; absolute encoders report a unique position value at all times."),
    ("Sensing", "IMU (Inertial Measurement Unit)", "An IMU typically combines an accelerometer (linear acceleration) and gyroscope (angular velocity), and sometimes a magnetometer (heading), used together for orientation estimation and motion tracking in mobile robots and drones."),
    ("Sensing", "Ultrasonic Distance Sensor", "Ultrasonic sensors measure distance by timing an emitted sound pulse's echo return, offering low-cost obstacle detection but limited accuracy on soft/angled surfaces compared to optical methods."),
    ("Sensing", "LiDAR Basics", "LiDAR measures distance by timing reflected laser pulses (or measuring phase shift), producing precise point clouds used for mapping, obstacle avoidance, and localization in mobile robotics and autonomous vehicles."),
    ("Sensing", "Wheel Odometry", "Wheel odometry estimates robot position by integrating wheel rotation counts over time, which is simple but accumulates drift error from wheel slip and needs periodic correction from other sensors (IMU, vision, LiDAR)."),
    ("Navigation", "Path Planning Overview", "Path planning algorithms compute a route from a start to a goal while avoiding obstacles. Common approaches include grid-search methods (A*, Dijkstra) and sampling-based methods (RRT, PRM) for higher-dimensional configuration spaces."),
    ("Navigation", "A* Search Algorithm", "A* finds a shortest path on a graph/grid by combining the actual cost so far with a heuristic estimate of remaining cost, expanding the most promising nodes first — widely used in grid-based robot path planning."),
    ("Navigation", "SLAM (Simultaneous Localization and Mapping)", "SLAM lets a robot build a map of an unknown environment while simultaneously tracking its own position within that map, typically fusing LiDAR/camera data with odometry and probabilistic filtering (e.g., particle filters, Kalman filters)."),
    ("Navigation", "Kalman Filter Basics", "A Kalman filter fuses noisy sensor measurements with a motion model to produce an optimal estimate of a system's state (e.g., position, velocity), widely used in robot localization and sensor fusion."),
    ("Mechanisms", "End Effectors and Grippers", "End effectors are the tools mounted at a robot arm's tip — parallel-jaw grippers, vacuum suction cups, and specialized tool changers are common choices depending on the object being manipulated."),
    ("Mechanisms", "Differential Drive Robot", "A differential drive robot steers by varying the relative speed of two independently driven wheels (often with passive casters for stability), making it mechanically simple compared to Ackermann or omnidirectional drive systems."),
    ("Software", "ROS (Robot Operating System) Overview", "ROS provides a middleware framework of nodes, topics, and services for building modular robotics software, letting sensor drivers, control algorithms, and planning modules communicate without being tightly coupled."),
    ("Software", "Microcontroller vs Single-Board Computer", "Microcontrollers (e.g., an ATmega or STM32) are efficient for real-time, low-level control (motors, sensors), while single-board computers (e.g., Raspberry Pi) suit higher-level tasks like vision processing and planning that benefit from a full OS."),
]

# ---------------------------------------------------------------------------
# PCB DESIGN
# ---------------------------------------------------------------------------
pcb_records = [
    ("Basics", "PCB Layer Stack-up", "A PCB stack-up defines how copper and dielectric layers are arranged through the board's thickness. A simple 2-layer board has top/bottom copper; 4+ layer boards add internal power/ground planes for better signal integrity and EMI performance."),
    ("Basics", "Traces and Copper Weight", "A PCB trace is a copper conductor connecting components. Copper weight (commonly 1oz or 2oz per square foot) and trace width together determine current-carrying capacity and resistance/voltage drop along the trace."),
    ("Basics", "Vias", "A via is a plated hole connecting copper on different layers. Through-hole vias span the full board; blind vias connect an outer layer to an inner layer; buried vias connect only internal layers, used in denser boards."),
    ("Basics", "Pads and Footprints", "A footprint defines the copper pad pattern, silkscreen outline, and mechanical keep-out for a specific component package, ensuring the physical part will solder correctly and fit as intended."),
    ("Basics", "Silkscreen Layer", "The silkscreen layer prints reference designators, polarity marks, and outlines onto the board surface for assembly and debugging, and carries no electrical function."),
    ("Basics", "Solder Mask", "Solder mask is the protective coating (commonly green, though many colors exist) that covers copper areas not meant to be soldered, preventing bridges and oxidation while leaving pads exposed."),
    ("Component Packages", "Through-Hole vs Surface Mount", "Through-hole components mount via leads passing through drilled holes, offering mechanical strength for connectors and large parts. Surface-mount (SMT) components solder directly onto pads, enabling smaller size and automated assembly."),
    ("Component Packages", "SOIC Package", "Small Outline Integrated Circuit (SOIC) is a common surface-mount IC package with gull-wing leads on two sides, easy to hand-solder and widely used for op-amps, logic ICs, and small microcontrollers."),
    ("Component Packages", "QFN Package", "Quad Flat No-lead (QFN) packages have pads flush with the package bottom (plus often a thermal pad), giving a compact, low-profile footprint but requiring reflow soldering and good thermal via design under the package."),
    ("Component Packages", "BGA Package", "Ball Grid Array (BGA) packages place solder balls in a grid under the chip, enabling very high pin counts in a small footprint, but requiring precise reflow profiles and X-ray inspection since joints are hidden underneath."),
    ("Component Packages", "0402 / 0603 / 0805 Passives", "These numeric codes describe the physical size of chip resistors/capacitors in hundredths of an inch (e.g., 0603 = 0.06in x 0.03in). Smaller codes save space but are harder to hand-solder and rework."),
    ("Signal Integrity", "Controlled Impedance Traces", "Controlled impedance design sets trace width, copper thickness, and dielectric spacing to hit a target characteristic impedance (commonly 50 ohm single-ended or 90/100 ohm differential), critical for high-speed digital and RF signals."),
    ("Signal Integrity", "Differential Pairs", "Differential signaling sends a signal and its inverse on two closely coupled traces; the receiver reads the difference between them, which rejects common-mode noise and is used for USB, Ethernet, and other high-speed interfaces."),
    ("Signal Integrity", "Ground Planes", "A continuous ground plane provides a low-inductance return path for signal currents, reduces EMI, and improves signal integrity — splitting or interrupting a ground plane under high-speed traces can create unwanted return-path detours."),
    ("Signal Integrity", "Crosstalk", "Crosstalk is unwanted coupling of a signal onto an adjacent trace via mutual capacitance/inductance. It's reduced by increasing trace spacing, adding ground traces between sensitive lines, and minimizing parallel trace length."),
    ("Power Integrity", "Power Plane Design", "A dedicated power plane layer distributes supply voltage with low impedance and inductance to all consuming components, and works with decoupling capacitors to keep the rail stable during fast current transients."),
    ("Power Integrity", "Decoupling Capacitor Placement", "Decoupling capacitors should be placed as close as physically possible to an IC's power pins, with short, wide traces or direct via connections, so their effectiveness at suppressing high-frequency noise isn't reduced by added inductance."),
    ("Thermal", "Thermal Vias", "An array of small vias under a hot component (e.g., a power regulator or QFN thermal pad) conducts heat from the top copper to an internal or bottom-layer copper pour/plane, improving heat dissipation."),
    ("Thermal", "Copper Pour for Heat Spreading", "A large copper pour connected to a component's ground/thermal pad acts as a heat spreader, lowering the component's effective thermal resistance to the surrounding air compared to a minimal-copper connection."),
    ("EMI/EMC", "EMI Basics", "Electromagnetic interference (EMI) is unwanted electrical noise radiated or conducted from a circuit. Good PCB layout (short loop areas, solid ground planes, proper decoupling) is often the cheapest way to reduce EMI before adding shielding/filtering."),
    ("EMI/EMC", "Loop Area Minimization", "Minimizing the physical loop area formed by a signal and its return path reduces both radiated emissions and susceptibility to external interference, since loop area is directly proportional to radiated/received EMI."),
    ("Manufacturing", "Gerber Files", "Gerber (RS-274X) files are the industry-standard format describing each copper layer, solder mask, silkscreen, and drill data needed for a fab house to manufacture a PCB."),
    ("Manufacturing", "Design for Manufacturing (DFM)", "DFM rules — minimum trace/space width, minimum drill size, annular ring size, and component spacing — ensure a design can be reliably fabricated and assembled at a given fab house's process capability."),
    ("Manufacturing", "Design for Assembly (DFA)", "DFA considers component orientation consistency, adequate spacing for pick-and-place equipment, and avoiding tall parts near dense fine-pitch ICs, so the board can be assembled reliably and cost-effectively."),
    ("Manufacturing", "Design Rule Check (DRC)", "A DRC automatically verifies a PCB layout against manufacturing and electrical rules (clearances, trace widths, unconnected nets) before files are sent to fabrication, catching errors early."),
    ("Tools", "Schematic Capture", "Schematic capture is the process of drawing a circuit's logical connections (as opposed to physical layout) using symbols for each component, which is then used to generate a netlist for PCB layout."),
    ("Tools", "Netlist", "A netlist is the list of electrical connections between component pins derived from a schematic; PCB layout tools use it to verify all required connections exist as traces are routed."),
]

# ---------------------------------------------------------------------------
# DIGITAL LOGIC DESIGN
# ---------------------------------------------------------------------------
digital_logic_records = [
    ("Number Systems", "Binary Number System", "Binary uses only digits 0 and 1, with each position representing a power of two. It's the native representation for digital logic since two voltage levels map naturally to two states."),
    ("Number Systems", "Hexadecimal Number System", "Hexadecimal (base 16, digits 0-9 and A-F) compactly represents binary values since each hex digit corresponds exactly to 4 bits, making it a common shorthand for memory addresses and register values."),
    ("Number Systems", "Binary to Decimal Conversion", "To convert binary to decimal, multiply each bit by its positional power of two and sum the results — e.g., binary 1011 = (1x8)+(0x4)+(1x2)+(1x1) = 11 decimal."),
    ("Number Systems", "Two's Complement", "Two's complement represents signed integers so that the same binary addition circuit works for both positive and negative numbers; negating a value is done by inverting all bits and adding 1."),
    ("Boolean Algebra", "Basic Boolean Laws", "Core Boolean identities include the identity law (A+0=A, A*1=A), null law (A+1=1, A*0=0), idempotent law (A+A=A), and complement law (A+A'=1, A*A'=0), used to simplify logic expressions."),
    ("Boolean Algebra", "De Morgan's Theorems", "De Morgan's theorems state that NOT(A AND B) = (NOT A) OR (NOT B), and NOT(A OR B) = (NOT A) AND (NOT B) — essential for converting between gate types and simplifying expressions."),
    ("Boolean Algebra", "Karnaugh Maps (K-Maps)", "A Karnaugh map is a grid-based visual method for simplifying Boolean expressions by grouping adjacent 1s (or 0s) in powers of two, reducing a truth table to a minimal sum-of-products (or product-of-sums) expression."),
    ("Boolean Algebra", "Sum of Products (SOP) and Product of Sums (POS)", "SOP form expresses a logic function as an OR of AND terms (each term true for one output-1 row); POS form expresses it as an AND of OR terms — both are standard canonical forms derived from a truth table."),
    ("Logic Gates", "AND Gate", "An AND gate outputs 1 only when all inputs are 1. Truth table for 2 inputs: 0,0->0; 0,1->0; 1,0->0; 1,1->1."),
    ("Logic Gates", "OR Gate", "An OR gate outputs 1 when at least one input is 1. Truth table for 2 inputs: 0,0->0; 0,1->1; 1,0->1; 1,1->1."),
    ("Logic Gates", "NOT Gate (Inverter)", "A NOT gate outputs the logical complement of its single input: 0->1 and 1->0. It's the simplest gate and is used to build all other gate types in CMOS logic."),
    ("Logic Gates", "NAND Gate", "A NAND gate outputs the inverse of AND: it outputs 0 only when all inputs are 1, otherwise 1. NAND is functionally complete — any Boolean function can be built using only NAND gates."),
    ("Logic Gates", "NOR Gate", "A NOR gate outputs the inverse of OR: it outputs 1 only when all inputs are 0, otherwise 0. Like NAND, NOR is functionally complete on its own."),
    ("Logic Gates", "XOR Gate", "An XOR (exclusive OR) gate outputs 1 when its inputs differ. Truth table for 2 inputs: 0,0->0; 0,1->1; 1,0->1; 1,1->0. XOR is central to binary addition and parity checking."),
    ("Logic Gates", "XNOR Gate", "An XNOR gate outputs 1 when its inputs are equal (the complement of XOR). Truth table for 2 inputs: 0,0->1; 0,1->0; 1,0->0; 1,1->1. It's commonly used for equality comparison."),
    ("Combinational Circuits", "Half Adder", "A half adder takes two single bits and outputs their Sum (via XOR) and Carry (via AND), but doesn't account for a carry-in from a previous stage."),
    ("Combinational Circuits", "Full Adder", "A full adder extends the half adder by also accepting a carry-in bit, producing Sum and Carry-out for that bit position — chaining full adders builds a ripple-carry multi-bit adder."),
    ("Combinational Circuits", "Multiplexer (MUX)", "A multiplexer selects one of several data inputs to route to a single output based on select lines — an N-to-1 mux needs log2(N) select lines. It's the digital equivalent of a rotary switch."),
    ("Combinational Circuits", "Demultiplexer (DEMUX)", "A demultiplexer routes a single input signal to one of several outputs based on select lines, performing the inverse operation of a multiplexer."),
    ("Combinational Circuits", "Encoder", "An encoder converts one of 2^N active input lines into an N-bit binary code representing which input was active, compressing many lines down to a compact code."),
    ("Combinational Circuits", "Decoder", "A decoder converts an N-bit binary input into one of 2^N active output lines, commonly used for memory address decoding or driving one-of-many enable signals."),
    ("Combinational Circuits", "Magnitude Comparator", "A magnitude comparator compares two multi-bit binary numbers and outputs whether one is greater than, less than, or equal to the other, typically built from cascaded single-bit comparison logic."),
    ("Sequential Circuits", "SR Latch", "An SR (Set-Reset) latch is the simplest memory element, built from cross-coupled NAND or NOR gates, holding its state until a Set or Reset input forces it high or low respectively — with an invalid state when both inputs are asserted simultaneously in the basic NOR version."),
    ("Sequential Circuits", "D Flip-Flop", "A D flip-flop captures the value on its D input at a clock edge and holds it until the next active edge, making it the standard building block for registers and pipeline stages."),
    ("Sequential Circuits", "JK Flip-Flop", "A JK flip-flop generalizes the SR latch behavior with a defined toggle state when both J and K inputs are 1, avoiding the invalid state problem of a basic SR latch."),
    ("Sequential Circuits", "T Flip-Flop", "A T (toggle) flip-flop changes state every clock edge when its T input is 1, and holds state when T is 0 — commonly used as the building block for binary counters."),
    ("Sequential Circuits", "Edge-Triggered vs Level-Triggered", "Edge-triggered flip-flops sample their input only at a clock transition (rising or falling edge), while level-triggered latches are transparent (output follows input) whenever the enable/clock is at the active level."),
    ("Sequential Circuits", "Shift Register", "A shift register chains flip-flops so data shifts one position per clock cycle, used for serial-to-parallel or parallel-to-serial data conversion and simple delay lines."),
    ("Sequential Circuits", "Binary Counter", "A binary counter is a chain of flip-flops (often T or JK type) that increments its stored binary value by one on each clock cycle, used for timing, frequency division, and sequencing."),
    ("Sequential Circuits", "Finite State Machine (FSM)", "An FSM models sequential logic as a set of defined states with transitions triggered by inputs, described as Moore (outputs depend only on state) or Mealy (outputs depend on state and input) machines."),
    ("Logic Families", "TTL Logic", "Transistor-Transistor Logic (TTL) is a bipolar-transistor-based logic family (5V nominal) historically dominant in discrete logic ICs, now largely superseded by CMOS for new designs due to CMOS's lower power consumption."),
    ("Logic Families", "CMOS Logic", "Complementary Metal-Oxide-Semiconductor (CMOS) logic uses paired PMOS/NMOS transistors so that, ideally, current flows only during switching transitions, giving very low static power consumption and making it the dominant logic family today."),
    ("Logic Families", "Logic Level Voltages", "Each logic family defines voltage ranges for a valid '0' and valid '1' with a forbidden/undefined region between them (noise margin), and mixing families (e.g., 3.3V and 5V logic) without level shifting can cause unreliable operation or damage."),
    ("Design Practice", "Combinational vs Sequential Logic", "Combinational logic output depends only on current inputs (no memory), while sequential logic output depends on current inputs and stored state from previous clock cycles, requiring flip-flops or latches."),
    ("Design Practice", "Setup and Hold Time", "Setup time is the minimum interval a data input must be stable before a clock edge; hold time is the minimum interval it must remain stable after that edge — violating either risks metastability or incorrect capture."),
    ("Design Practice", "Clock Domain Crossing", "When a signal moves between two circuits driven by different (asynchronous) clocks, synchronizer flip-flops are needed to reduce the risk of metastability propagating into the receiving clock domain."),
    ("HDL", "Verilog Basics", "Verilog is a hardware description language used to model digital logic at behavioral, dataflow, or structural levels of abstraction, then synthesized into actual gate-level implementations for an FPGA or ASIC."),
    ("HDL", "VHDL Basics", "VHDL is a strongly-typed hardware description language, often preferred in defense/aerospace and European industry contexts, describing digital circuits through entities (interfaces) and architectures (behavior)."),
    ("HDL", "Testbenches", "A testbench is HDL code written purely for simulation — not synthesized into hardware — that applies stimulus to a design under test and checks its outputs against expected behavior before committing to fabrication or FPGA deployment."),
]


def build_rows():
    rows = []
    idx = 1

    def add(category, records):
        nonlocal idx
        for subcat, title, content in records:
            rows.append({
                "id": idx,
                "category": category,
                "subcategory": subcat,
                "title": title,
                "content": content,
                "tags": f"{category.lower()};{subcat.lower().replace(' ', '_')}",
            })
            idx += 1

    add("Electronics", electronics_records)
    add("Robotics", robotics_records)
    add("PCB Design", pcb_records)
    add("Digital Logic Design", digital_logic_records)
    return rows


def load_extra_sources():
    """
    Optional: ingest your own additional records to push past 1,000 rows.

    Drop a CSV into data/raw_sources/ with columns:
        category, subcategory, title, content
    and it will be picked up automatically and appended (id auto-assigned).
    """
    extra = []
    raw_dir = os.path.join(OUT_DIR, "raw_sources")
    if not os.path.isdir(raw_dir):
        return extra
    for fname in sorted(os.listdir(raw_dir)):
        if not fname.lower().endswith(".csv"):
            continue
        path = os.path.join(raw_dir, fname)
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                extra.append((
                    row.get("category", "Uncategorized"),
                    row.get("subcategory", ""),
                    row.get("title", ""),
                    row.get("content", ""),
                ))
    return extra


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = build_rows()

    # Fold in any extra CSVs the user has dropped into data/raw_sources/
    extras = load_extra_sources()
    next_id = len(rows) + 1
    for category, subcat, title, content in extras:
        rows.append({
            "id": next_id,
            "category": category,
            "subcategory": subcat,
            "title": title,
            "content": content,
            "tags": f"{category.lower()};{subcat.lower().replace(' ', '_')}",
        })
        next_id += 1

    for row in rows:
        row.setdefault("source_url", "https://github.com/munsakad/DMHardware-Chat-Bot/blob/main/generate_dataset.py")
        row.setdefault("source_group", "original:" + row["title"])
        row.setdefault("record_type", "original_reference")
    rows.extend(expanded_records())
    unique = {}
    for row in rows:
        key = " ".join(row["content"].split()).lower()
        if key and key not in unique:
            unique[key] = row
    rows = list(unique.values())
    for index, row in enumerate(rows, 1):
        row["id"] = index
    assert len(rows) >= 1000, f"Only {len(rows)} unique records"
    manifest = {"records": len(rows), "categories": dict(Counter(r["category"] for r in rows)),
                "record_types": dict(Counter(r["record_type"] for r in rows)),
                "source_lessons_commit": "c5fa04da7f8be79333d44a78d2cda816b7c94075",
                "source_lessons_repository": "munsakad/Hardware-Engineering-Education-Hub",
                "note": "Counts are retrieval records, not independent knowledge areas or measured observations."}
    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "category", "subcategory", "title", "content", "tags", "source_url", "source_group", "record_type"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} records to {OUT_PATH}")


if __name__ == "__main__":
    main()
