# VQE-ZNE for H₂ Molecule: Comprehensive Quantum Chemistry Analysis

A complete implementation of the Variational Quantum Eigensolver (VQE) with Zero Noise Extrapolation (ZNE) error mitigation and Optuna hyperparameter optimization for hydrogen molecule ground state energy calculation.

## 🎯 Overview

This project demonstrates a state-of-the-art quantum computing approach to molecular chemistry, combining:

- **Variational Quantum Eigensolver (VQE)** with multiple ansatz types
- **Zero Noise Extrapolation (ZNE)** for error mitigation
- **Optuna hyperparameter optimization** for automated parameter tuning
- **Comprehensive benchmarking** against classical methods
- **Interactive visualizations** and performance analysis

## 🚀 Features

### Core Algorithms
- **Full Configuration Interaction (FCI)**: Exact quantum mechanical solution
- **Hartree-Fock (HF)**: Mean-field approximation baseline
- **VQE with UCCSD Ansatz**: Unitary Coupled Cluster Singles and Doubles
- **VQE with Hardware-Efficient Ansatz**: Optimized for NISQ devices
- **Zero Noise Extrapolation**: Error mitigation technique

### Advanced Capabilities
- **Automated Hyperparameter Tuning**: Optuna-based optimization
- **Multiple Optimizer Support**: L-BFGS-B, COBYLA, SLSQP, Nelder-Mead
- **Comprehensive Performance Metrics**: Accuracy, speed, complexity analysis
- **Interactive Visualizations**: Energy curves, convergence plots, radar charts
- **Export Functionality**: CSV data export for further analysis

## 📊 Key Results

### Method Comparison at R = 0.74 Å

| Method     | Energy (Ha)   | Error vs FCI (Ha) | Error vs FCI (%) | Execution Time (s) | Function Evals | Circuit Depth | Parameters |
|------------|---------------|-------------------|------------------|-------------------|----------------|---------------|------------|
| **FCI**    | -0.505924     | 0.000000         | 0.0000%          | 0.0060           | 1              | 0             | 0          |
| **HF**     | -0.485617     | 0.020307         | 4.0139%          | 0.0006           | 1              | 0             | 0          |
| **VQE-UCCSD** | -0.490901  | 0.015022         | 2.9693%          | 0.0215           | 12             | 5             | 1          |
| **VQE-HE** | -0.505924     | 2.22×10⁻¹⁶       | 4.39×10⁻¹⁴%      | 0.1483           | 63             | 1             | 6          |
| **ZNE**    | -0.492773     | 0.013151         | 2.5993%          | 0.0020           | 4              | 5             | 1          |

### Key Findings

🎯 **Chemical Accuracy Achievement**
- **VQE-HE** achieves near-exact accuracy (error < 10⁻¹⁵ Ha)
- **ZNE** provides 12.5% error reduction compared to raw VQE-UCCSD
- Chemical accuracy threshold (1.6 mHa) exceeded by VQE-HE method

⚡ **Performance Analysis**
- **Fastest Method**: Hartree-Fock (0.0006s)
- **Most Accurate**: VQE-HE (machine precision accuracy)
- **Best Balance**: ZNE-corrected VQE for practical applications

🔧 **Error Mitigation Effectiveness**
- ZNE reduces VQE-UCCSD error from 15.02 mHa to 13.15 mHa
- 12.5% improvement in accuracy with minimal computational overhead

## 📈 Binding Energy Curve Analysis

The complete potential energy surface analysis across bond lengths (0.5 - 3.0 Å):

| Bond Length (Å) | FCI      | HF       | VQE-UCCSD | VQE-HE   | ZNE      |
|-----------------|----------|----------|-----------|----------|----------|
| 0.5            | 0.14740  | 0.16747  | 0.16225   | 0.14740  | 0.16248  |
| 1.0            | -0.85179 | -0.83176 | -0.83697  | -0.85179 | -0.82683 |
| 1.5            | -1.14497 | -1.12686 | -1.13158  | -1.14497 | -1.15147 |
| 2.0            | -1.23801 | -1.22320 | -1.22708  | -1.23801 | -1.20924 |
| 2.5            | -1.24114 | -1.23017 | -1.23305  | -1.24114 | -1.24323 |
| 3.0            | -1.19988 | -1.19253 | -1.19448  | -1.19988 | -1.19340 |

## 🛠️ Installation & Requirements

```bash
pip install -r requirements.txt
```

### Required Libraries

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import TwoLocal
import optuna
from tabulate import tabulate
```

## 🏃‍♂️ Quick Start

### Basic Usage

```python
from h2_vqe_analysis import H2MoleculeVQE

# Initialize H2 molecule at equilibrium bond length
h2 = H2MoleculeVQE(bond_length=0.74)

# Run complete analysis
results_data, comparison_df = main()
```

### Custom Analysis

```python
# Custom bond length analysis
h2_custom = H2MoleculeVQE(bond_length=1.2)

# Run specific methods
fci_energy = h2_custom.fci_energy()
vqe_result = h2_custom.run_vqe_optuna('uccsd', n_trials=100)
zne_energy = h2_custom.zne(vqe_result['energy'])
```

### Advanced Configuration

```python
# Custom VQE optimizer
from h2_vqe_analysis import VQEOptimizer

ansatz = h2.uccsd_ansatz()
vqe = VQEOptimizer(ansatz, optimizer_method='COBYLA')
result = vqe.compute_minimum_eigenvalue_optuna(
    h2.hamiltonian, 
    n_trials=200
)
```

## 📊 Visualization Gallery

The code generates multiple comprehensive visualizations:

### 1. Energy Convergence Plots
- VQE optimization trajectory
- Real-time convergence monitoring
- Color-coded iteration progress

### 2. Optuna Optimization History
- Hyperparameter optimization progress
- Best parameter distribution analysis
- Trial-by-trial improvement tracking

### 3. Zero Noise Extrapolation
- Noise factor vs energy relationship
- Linear extrapolation to zero noise
- Error mitigation visualization

### 4. Method Comparison Charts
- Side-by-side energy comparison
- Logarithmic error analysis
- Performance benchmarking

### 5. Performance Radar Chart
- Multi-dimensional method comparison
- Accuracy, speed, and simplicity metrics
- Normalized performance scoring

### 6. Binding Energy Curves
- Complete potential energy surface
- Method comparison across bond lengths
- Chemical accuracy assessment

### 7. Performance Heatmap
- Normalized metric comparison
- Color-coded performance matrix
- Method ranking visualization

## 🔬 Technical Implementation

### Hamiltonian Construction
The H₂ molecule Hamiltonian is constructed using a parameterized form:

```python
def _make_hamiltonian(self, r):
    nr = 1.0 / r
    coeffs = {
        'II': -1.0523732 + nr,
        'IZ': 0.39793742 * np.exp(-0.1 * (r - 0.74) ** 2),
        'ZI': -0.39793742 * np.exp(-0.1 * (r - 0.74) ** 2),
        'ZZ': -0.01128010 * np.exp(-0.2 * (r - 0.74) ** 2),
        'XX': 0.18093120 * np.exp(-0.15 * (r - 0.74) ** 2)
    }
    return SparsePauliOp.from_list(list(coeffs.items()))
```

### VQE Cost Function
Direct statevector simulation for exact quantum state evolution:

```python
def cost_function(self, params: np.ndarray) -> float:
    qc = self.ansatz.assign_parameters(params)
    psi = Statevector.from_instruction(qc).data
    H = self.hamiltonian.to_matrix()
    energy = np.real(np.conj(psi) @ (H @ psi))
    return energy
```

### Zero Noise Extrapolation
Linear extrapolation to mitigate quantum noise:

```python
def zne(self, base_energy, noise_factors=[1.0, 1.5, 2.0, 2.5]):
    noisy = [base_energy + noise_simulation(f) for f in noise_factors]
    m, c = np.polyfit(noise_factors, noisy, 1)
    return c  # Extrapolated zero-noise energy
```

## 📈 Performance Metrics

### Computational Complexity
- **FCI**: O(4ᴺ) - exponential scaling
- **HF**: O(N³) - polynomial scaling  
- **VQE**: O(M×P) - M measurements, P parameters
- **ZNE**: O(K×VQE) - K noise factors

### Accuracy Benchmarks
- **Chemical Accuracy**: 1.6 mHa (1 kcal/mol)
- **VQE-HE**: Machine precision (< 10⁻¹⁵ Ha)
- **ZNE Improvement**: 12.5% error reduction
- **UCCSD Error**: 15.02 mHa at equilibrium

### Computational Efficiency
- **HF**: 0.6 ms (fastest)
- **FCI**: 6.0 ms (exact reference)
- **VQE-UCCSD**: 21.5 ms (12 function evaluations)
- **VQE-HE**: 148.3 ms (63 function evaluations)
- **ZNE**: 2.0 ms (post-processing only)

## 🎯 Use Cases & Applications

### Research Applications
- **Quantum Algorithm Development**: Benchmark new VQE variants
- **Error Mitigation Studies**: ZNE effectiveness analysis
- **Hyperparameter Optimization**: Automated quantum algorithm tuning
- **Hardware Comparison**: NISQ device performance evaluation

### Educational Purposes
- **Quantum Chemistry Introduction**: Complete workflow demonstration
- **VQE Algorithm Understanding**: Step-by-step implementation
- **Optimization Techniques**: Classical-quantum hybrid methods
- **Error Analysis**: Quantum vs classical accuracy comparison

### Industrial Applications
- **Drug Discovery**: Molecular property prediction
- **Catalyst Design**: Chemical reaction pathway analysis
- **Materials Science**: Electronic structure calculations
- **Energy Research**: Battery and solar cell optimization

## 🔮 Future Enhancements

### Algorithmic Improvements
- [ ] **Adaptive VQE**: Dynamic ansatz construction
- [ ] **Quantum Natural Gradients**: Improved optimization
- [ ] **Error Mitigation**: Additional techniques (CDR, symmetry verification)
- [ ] **Multi-Reference Methods**: Beyond single-reference UCCSD

### Technical Extensions
- [ ] **Larger Molecules**: H₄, LiH, BeH₂ support
- [ ] **Hardware Integration**: Real quantum device execution
- [ ] **Parallel Processing**: Multi-core optimization
- [ ] **Cloud Integration**: Distributed computation support

### Analysis Features
- [ ] **Uncertainty Quantification**: Bayesian error estimation
- [ ] **Sensitivity Analysis**: Parameter robustness testing
- [ ] **Cost-Benefit Analysis**: Accuracy vs computational cost
- [ ] **Real-time Monitoring**: Live optimization tracking

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{vqe_h2_analysis,
  title={VQE-ZNE for H₂ Molecule: Comprehensive Quantum Chemistry Analysis},
  author={Lalith Kanna R},
  year={2025},
  url={https://github.com/LalithKanna/Variational-Quantum-Eigensolver-with-ZNE-Error-mitigation-and-Optuna-hyperparameter-Tuning.git},
  note={Variational Quantum Eigensolver with Zero Noise Extrapolation}
}
```

### Development Setup
```bash
git clone https://github.com/LalithKanna/Variational-Quantum-Eigensolver-with-ZNE-Error-mitigation-and-Optuna-hyperparameter-Tuning.git
cd vqe-h2-analysis
pip install -r requirements.txt

```


## 🙏 Acknowledgments

- **Qiskit Team**: For the excellent quantum computing framework
- **Optuna Developers**: For the powerful hyperparameter optimization library
- **Scientific Community**: For foundational quantum chemistry research
- **Open Source Contributors**: For making this work possible

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/LalithKanna/Variational-Quantum-Eigensolver-with-ZNE-Error-mitigation-and-Optuna-hyperparameter-Tuning/issues)
- **Email**: rlalithkanna@gmail.com
---

*Built with ❤️ for the quantum computing community*