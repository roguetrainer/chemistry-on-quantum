# VQE Simulation of the Hydrogen Molecule (H2)
# This script finds the ground state energy of H2 using the Variational Quantum Eigensolver (VQE).
# The molecular Hamiltonian generation uses PennyLane's quantum chemistry module, which builds upon
# concepts and mappings (like Jordan-Wigner) pioneered by projects like OpenFermion.

import pennylane as qml
from pennylane import numpy as np
import time

# --- 1. Define Molecular Parameters ---

# Define the atomic symbols and coordinates for H2 (at equilibrium bond length ~0.74 Angstroms or 1.4 Bohr)
symbols = ["H", "H"]
# Using a specific bond length (in Bohr)
geometry = np.array([0.0, 0.0, -0.6614, 0.0, 0.0, 0.6614]) 

# Quantum chemical parameters
charge = 0
multiplicity = 1 # Spin multiplicity (2S + 1). H2 ground state is a singlet (S=0, multiplicity=1)
basis_set = "sto-3g" # Minimal basis set
active_electrons = 2 # Total number of electrons
active_orbitals = 2  # Number of spatial orbitals

print("--- Step 1: Generating Hamiltonian ---")

# Use PennyLane's qchem to construct the qubit Hamiltonian.
# This function performs the electronic structure calculation,
# maps the fermionic Hamiltonian to a qubit Hamiltonian (via Jordan-Wigner by default),
# and returns the number of qubits required.
try:
    # Set method='openfermion' if you have the OpenFermion-PySCF plugin installed,
    # but the default method='qchem' is used here for stability.
    H, num_qubits = qml.qchem.molecular_hamiltonian(
        symbols,
        geometry,
        charge=charge,
        mult=multiplicity,
        basis=basis_set,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals
    )
    print(f"Number of qubits required: {num_qubits}")
    print(f"Hamiltonian terms: {len(H.ops)}")
except Exception as e:
    print(f"Error during Hamiltonian generation: {e}")
    print("Ensure you have all required PennyLane dependencies installed.")
    exit()

# Extract the Hartree-Fock (HF) state, which is the starting point for the VQE ansatz
hf_state = qml.qchem.hf_state(active_electrons, num_qubits)
print(f"Hartree-Fock state: {hf_state}")


# --- 2. Define the Quantum Device and Ansatz (Circuit) ---

# Define the device (a simulator)
dev = qml.device("default.qubit", wires=num_qubits)

# Define the Unitary Coupled-Cluster Singles and Doubles (UCCSD) ansatz.
# For H2 in the minimal basis (4 spin orbitals, 2 electrons), the only non-zero
# excitation is a single double-excitation, making this the optimal ansatz.
singles, doubles = qml.qchem.excitations(active_electrons, num_qubits)

def circuit(params, wires):
    """The variational circuit (Ansatz)."""
    # 1. Prepare the Hartree-Fock reference state
    qml.BasisState(hf_state, wires=wires)
    
    # 2. Apply the UCCSD ansatz
    qml.UCCSD(params, wires, singles=singles, doubles=doubles)

# Create the QNode (Quantum Node) that measures the Hamiltonian expectation value
@qml.qnode(dev, interface="autograd")
def cost_fn(params):
    """Returns the expectation value of the Hamiltonian."""
    # The circuit acts to prepare the trial ground state
    circuit(params, wires=range(num_qubits))
    
    # Measure the expectation value of the Hamiltonian
    return qml.expval(H)


# --- 3. Optimization and VQE Loop ---

print("\n--- Step 2: Running VQE Optimization ---")
start_time = time.time()

# Initialize the optimizer and parameters
# The UCCSD template requires a parameter for each excitation. For H2/STO-3G, this is one double-excitation.
param_shape = qml.UCCSD.shape(active_electrons, num_qubits, singles, doubles)
params = np.random.normal(0, np.pi, param_shape, requires_grad=True)

# Use a classical optimizer (Adaptive Moment Estimation)
opt = qml.AdamOptimizer(stepsize=0.01)

# Optimization settings
max_iterations = 100
convergence_tolerance = 1e-06

# Run the optimization loop
energy_history = []
for n in range(max_iterations):
    # Perform one optimization step and calculate the new energy
    params, prev_energy = opt.step_and_cost(cost_fn, params)
    
    energy = cost_fn(params)
    energy_history.append(energy)
    
    conv = np.abs(energy - prev_energy)
    
    if n % 20 == 0:
        print(f"Iteration = {n:3d}, Energy = {energy:.8f} Ha, Convergence = {conv:.8f}")

    if conv <= convergence_tolerance:
        break

end_time = time.time()

# --- 4. Results ---

print("\n--- Step 3: Results ---")
print(f"Final ground state energy: {energy:.8f} Ha")
print(f"Optimal parameters: {params}")
print(f"Total optimization steps: {n + 1}")
print(f"Execution time: {end_time - start_time:.2f} seconds")

# Classically computed Full Configuration Interaction (FCI) energy for comparison
# FCI energy for H2 at this bond length is approximately -1.13726 Hartree
fci_energy = -1.13726
print(f"Reference FCI energy:     {fci_energy:.8f} Ha")

if np.abs(energy - fci_energy) < 1e-04:
    print("\nSUCCESS: VQE result is chemically accurate!")
else:
    print("\nNOTE: VQE converged, but check parameters/ansatz if accuracy is low.")

# Optional: Plot the optimization history
# import matplotlib.pyplot as plt
# plt.plot(energy_history, label="VQE Energy")
# plt.axhline(fci_energy, color='r', linestyle='--', label="FCI Energy")
# plt.xlabel("Optimization Step")
# plt.ylabel("Energy (Hartree)")
# plt.title("VQE Optimization for H2 Ground State")
# plt.legend()
# plt.show()
