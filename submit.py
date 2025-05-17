
import networkx as nx
from networkx.generators.degree_seq import random_degree_sequence_graph
import numpy as np
import pandas as pd
import time

from typing import List
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

import random


def ring_mesh_8():
    G = nx.Graph()
    G.add_edges_from([(i, (i + 1) % 8) for i in range(8)])
    return G

def bi_complete_8x8():
    left = list(range(8))
    right = list(range(8, 16))
    G = nx.Graph()
    G.add_nodes_from(left + right)

    for u in left:
        for v in right:
            G.add_edge(u, v)
    return G

def bi_complete_nxn(n):
    left = list(range(n))
    right = list(range(n, 2 * n))
    G = nx.Graph()
    G.add_nodes_from(left + right)
    for u in left:
        for v in right:
            G.add_edge(u, v)
    return G

def reg_graph_8():
    seq = [4] * 8
    tries = 0
    while True:
        G = random_degree_sequence_graph(seq, seed=random.randint(0, 1000))

        if nx.is_connected(G) and all(d == 4 for _, d in G.degree()):
            break
        tries += 1
        if tries > 10:
            break
    return G

def cubic_graph_16():
    G = nx.random_regular_graph(3, 16, seed=1234)
    return G

def random_connected_graph_16(p=0.18):
    n = 16
    tries = 0
    while True:
        G = nx.erdos_renyi_graph(n, p, seed=random.randint(0, 10000))
        if nx.is_connected(G):
            break
        tries += 1
        if tries > 20:
            break
    return G

def expander_graph_n(n):
    G = nx.random_regular_graph(4, n, seed=99)
    return G

def defective_grid_4x4():
    G = nx.Graph()
    G.add_nodes_from(range(16))

    for i in range(16):
        if i % 4 != 3 and not (i == 5):
            G.add_edge(i, i + 1)

    for i in range(12):
        G.add_edge(i, i + 4)

    return G

graph1 = ring_mesh_8()
graph2 = bi_complete_8x8()
graph3 = bi_complete_nxn(5)
graph4 = reg_graph_8()
graph5 = cubic_graph_16()
graph6 = random_connected_graph_16(p=0.18)
graph7 = expander_graph_n(16)
graph8 = defective_grid_4x4()

graph = graph1

#####################################################
# You can edit the code below this line!       #
#####################################################


num_steps=20 #number of QITE steps
lr=0.1 #learning rate

def build_ansatz(graph: nx.Graph) -> QuantumCircuit:
    ansatz = QuantumCircuit(graph.number_of_nodes())
    ansatz.h(range(graph.number_of_nodes()))
    num_rounds = 2

    for round in range(num_rounds):
        first_round = []

        random_root = random.choice(list(graph.nodes))
        graph_bfs = nx.bfs_tree(graph, source=random_root)
        theta = ParameterVector(fr"$\theta_{round}$", graph_bfs.number_of_edges())
        for t, (u, v) in zip(theta, graph_bfs.edges):
            first_round.append((u, v))
            ansatz.s(v)
            ansatz.h(v)
            ansatz.cx(u, v)
            ansatz.rz(t, v)
            ansatz.cx(u, v)
            ansatz.h(v)
            ansatz.sdg(v)

        graph_copy = graph.copy()
        for u, v in graph_bfs.edges:
            graph_copy.remove_edge(u, v)
        graph_copy.remove_nodes_from(list(nx.isolates(graph_copy)))
        random_root = random.choice(list(graph_copy.nodes))
        graph_copy_bfs = nx.bfs_tree(graph_copy, source=random_root)
        phi = ParameterVector(fr"$\phi_{round}$", graph_copy_bfs.number_of_edges())
        for p, (u, v) in zip(phi, graph_copy_bfs.edges):
            first_round.append((u, v))
            ansatz.s(v)
            ansatz.h(v)
            ansatz.cx(u, v)
            ansatz.rz(p, v)
            ansatz.cx(u, v)
            ansatz.h(v)
            ansatz.sdg(v)

        eta = ParameterVector(fr"$\eta_{round}$", graph_bfs.number_of_edges())
        for t, (u, v) in zip(eta, first_round):
            ansatz.s(u)
            ansatz.h(u)
            ansatz.cx(v, u)
            ansatz.rz(t, u)
            ansatz.cx(v, u)
            ansatz.h(u)
            ansatz.sdg(u)

    return ansatz

def build_maxcut_hamiltonian(graph: nx.Graph) -> SparsePauliOp:
    """
    Build the MaxCut Hamiltonian for the given graph H = (|E|/2)*I - (1/2)*Σ_{(i,j)∈E}(Z_i Z_j)
    """
    num_qubits = len(graph.nodes)
    edges = list(graph.edges())
    num_edges = len(edges)

    pauli_terms = ["I"*num_qubits] # start with identity
    coeffs = [-num_edges / 2]

    for (u, v) in edges: # for each edge, add -(1/2)*Z_i Z_j
        z_term = ["I"] * num_qubits
        z_term[u] = "Z"
        z_term[v] = "Z"
        pauli_terms.append("".join(z_term))
        coeffs.append(0.5)

    return SparsePauliOp.from_list(list(zip(pauli_terms, coeffs)))

class QITEvolver:
    """
    A class to evolve a parametrized quantum state under the action of an Ising
    Hamiltonian according to the variational Quantum Imaginary Time Evolution
    (QITE) principle described in IonQ's latest joint paper with ORNL.
    """
    def __init__(self, hamiltonian: SparsePauliOp, ansatz: QuantumCircuit):
        self.hamiltonian = hamiltonian
        self.ansatz = ansatz

        # Define some constants
        self.backend = AerSimulator()
        self.num_shots = 10000.0
        self.energies, self.param_vals, self.runtime = list(), list(), list()

    def get_defining_ode(self, measurements: List[dict[str, int]]):
        """
        Construct the dynamics matrix and load vector defining the varQITE
        iteration.
        """
        # Load sampled bitstrings and corresponding frequencies into NumPy arrays
        dtype = np.dtype([("states", int, (self.ansatz.num_qubits,)), ("counts", "f")])
        measurements = [np.fromiter(map(lambda kv: (list(kv[0]), kv[1]), res.items()), dtype) for res in measurements]

        # Set up the dynamics matrix by computing the gradient of each Pauli word
        # with respect to each parameter in the ansatz using the parameter-shift rule
        pauli_terms = [SparsePauliOp(op) for op, _ in self.hamiltonian.label_iter() if set(op) != set("I")]
        Gmat = np.zeros((len(pauli_terms), self.ansatz.num_parameters))
        for i, pauli_word in enumerate(pauli_terms):
            for j, jth_pair in enumerate(zip(measurements[1::2], measurements[2::2])):
                for pm, pm_shift in enumerate(jth_pair):
                    Gmat[i, j] += (-1)**pm * expected_energy(pauli_word, pm_shift)

        # Set up the load vector
        curr_energy = expected_energy(self.hamiltonian, measurements[0])
        dvec = np.zeros(len(pauli_terms))
        for i, pauli_word in enumerate(pauli_terms):
            rhs_op_energies = get_ising_energies(pauli_word, measurements[0]["states"])
            rhs_op_energies *= get_ising_energies(self.hamiltonian, measurements[0]["states"]) - curr_energy
            dvec[i] = -np.dot(rhs_op_energies, measurements[0]["counts"]) / self.num_shots
        return Gmat, dvec, curr_energy

    def get_iteration_circuits(self, curr_params: np.array):
        """
        Get the bound circuits that need to be evaluated to step forward
        according to QITE.
        """
        # Use this circuit to estimate your Hamiltonian's expected value
        circuits = [self.ansatz.assign_parameters(curr_params)]

        # Use these circuits to compute gradients
        for k in np.arange(curr_params.shape[0]):
            for j in range(2):
                pm_shift = curr_params.copy()
                pm_shift[k] += (-1)**j * np.pi/2
                circuits += [self.ansatz.assign_parameters(pm_shift)]

        # Add measurement gates and return
        [qc.measure_all() for qc in circuits]
        return circuits

    def print_status(self, measurements):
        """
        Print summary statistics describing a QITE run.
        """
        stats = pd.DataFrame({
            "curr_energy": self.energies,
            "num_circuits": [len(measurements)] * len(self.energies),
            "quantum_exec_time": self.runtime
        })
        stats.index.name = "step"
        print(stats)

    def evolve(self, num_steps: int, lr: float = 0.4):
        """
        Evolve the variational quantum state encoded by ``self.ansatz`` under
        the action of ``self.hamiltonian`` according to varQITE.
        """
        curr_params = np.zeros(self.ansatz.num_parameters)
        for k in range(num_steps):
            # Get circuits and measure on backend
            iter_qc = self.get_iteration_circuits(curr_params)
            job = self.backend.run(iter_qc, shots=self.num_shots)
            q0 = time.time()
            measurements = job.result().get_counts()
            quantum_exec_time = time.time() - q0

            # Update parameters-- set up defining ODE and step forward
            Gmat, dvec, curr_energy = self.get_defining_ode(measurements)
            dcurr_params = np.linalg.lstsq(Gmat, dvec, rcond=1e-2)[0]
            curr_params += lr * dcurr_params

            # Progress checkpoint!
            self.energies.append(curr_energy)
            self.param_vals.append(curr_params.copy())
            self.runtime.append(quantum_exec_time)

#####################################################
# Do not modify the code below this line!       #
#####################################################
            print(k, return_results(self))

def return_results(qit_evolver):
    """
    Return the results of the QITE run.
    """
    return {
        "energies": qit_evolver.energies[-1],
        "runtime": qit_evolver.runtime[-1]
    }

##### Utility functions #####
def compute_cut_size(graph, bitstring):
    """
    Get the cut size of the partition of ``graph`` described by the given
    ``bitstring``.
    """
    cut_sz = 0
    for (u, v) in graph.edges:
        if bitstring[u] != bitstring[v]:
            cut_sz += 1
    return cut_sz

def get_ising_energies(operator: SparsePauliOp, states: np.array):
    """
    Get the energies of the given Ising ``operator`` that correspond to the
    given ``states``.
    """
    # Unroll Hamiltonian data into NumPy arrays
    paulis = np.array([list(ops) for ops, _ in operator.label_iter()]) != "I"
    coeffs = operator.coeffs.real

    # Vectorized energies computation
    energies = (-1) ** (states @ paulis.T) @ coeffs
    return energies

def expected_energy(hamiltonian: SparsePauliOp, measurements: np.array):
    """
    Compute the expected energy of the given ``hamiltonian`` with respect to
    the observed ``measurement``.

    The latter is assumed to by a NumPy records array with fields ``states``
    --describing the observed bit-strings as an integer array-- and ``counts``,
    describing the corresponding observed frequency of each state.
    """
    energies = get_ising_energies(hamiltonian, measurements["states"])
    return np.dot(energies, measurements["counts"]) / measurements["counts"].sum()




ansatz = build_ansatz(graph)
ham = build_maxcut_hamiltonian(graph)
qit_evolver = QITEvolver(ham, ansatz)

qit_evolver.evolve(num_steps, lr)

from qiskit_aer import AerSimulator
shots = 100_000

# Sample your optimized quantum state using Aer
backend = AerSimulator()
optimized_state = ansatz.assign_parameters(qit_evolver.param_vals[-1])
optimized_state.measure_all()
counts = backend.run(optimized_state, shots=shots).result().get_counts()

# Find the sampled bitstring with the largest cut value
cut_vals = sorted(((bs, compute_cut_size(graph, bs)) for bs in counts), key=lambda t: t[1])
best_bs = cut_vals[-1][0]

print("\n\n\nbitstring\tcuts\tcounts")
print("----------------------------------")
for i in cut_vals:
    print(i[0],'\t', i[1], '\t', counts[i[0]])
print("\n\n\n")

################## Scoring ##################
G = graph
n = len(G.nodes())
w = np.zeros([n, n])
for i in range(n):
    for j in range(n):
        temp = G.get_edge_data(i, j, default=0)
        if temp != 0:
            w[i, j] = 1.0

best_cost_brute = 0

for b in range(2**n):
    x = [int(t) for t in reversed(list(bin(b)[2:].zfill(n)))]

    # Create subgraphs based on the partition
    subgraph0 = G.subgraph([i for i, val in enumerate(x) if val == 0])
    subgraph1 = G.subgraph([i for i, val in enumerate(x) if val == 1])

    bs = "".join(str(i) for i in x)

    # Check if subgraphs are not empty
    if len(subgraph0.nodes) > 0 and len(subgraph1.nodes) > 0:
        cost = 0
        for i in range(n):
            for j in range(n):
                cost = cost + w[i, j] * x[i] * (1 - x[j])
        if best_cost_brute < cost:
            best_cost_brute = cost
            xbest_brute = x
            XS_brut = []
        if best_cost_brute == cost:
            XS_brut.append(bs)

print("Best bitstrings: " + str(XS_brut))

def final_score(graph, optimal_bitstrings, counts, shots, ansatz):

    num_qubits = graph.number_of_nodes()
    inaccuracy, accuracy_score = 0, 0.0

    target_solutions = optimal_bitstrings
    # ensure optimal_bitstrings have the correct length
    optimal_bitstrings_formatted = [bs.zfill(num_qubits) for bs in target_solutions]

    theoretical_amplitude = int(shots/len(optimal_bitstrings_formatted))
    for bitstring in optimal_bitstrings_formatted:
        if bitstring in counts:
            inaccuracy += abs(counts[bitstring]-theoretical_amplitude)
        else:
            inaccuracy += theoretical_amplitude

    accuracy_score = 1.0 - inaccuracy/shots

    # CNOT count penalty
    try:
        transpiled_ansatz = transpile(ansatz, basis_gates=['cx', 'rz', 'sx', 'x'])
        ops_count = transpiled_ansatz.count_ops()
        cx_count = ops_count.get('cx', 0)
    except Exception as e:
        print(f"Error during transpilation or counting ops: {e}")
        cx_count = float('inf')

    num_edges = graph.number_of_edges()
    denominator = (8 * num_edges + cx_count)
    if denominator == 0 :
        efficiency_score = 1.0
    else:
        efficiency_score = (8 * num_edges) / denominator


    score = efficiency_score * accuracy_score

    return np.round(score, 5)

print("Final score: " + str(final_score(graph, XS_brut, counts, shots, ansatz)))


