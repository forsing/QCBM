# QCBM (Quantum Circuit Born Machine)

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from qiskit.visualization import circuit_drawer

from IPython.display import display
from IPython.display import clear_output


from qiskit_machine_learning.utils import algorithm_globals
import random

# =========================
# Seed za reproduktivnost
# =========================
SEED = 35
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED

# 1. Učitaj loto podatke
df = pd.read_csv("/Users/milan/Desktop/GHQ/data/loto5_89_k80.csv", header=None)


###################################
print()
print("Prvih 5 ucitanih kombinacija iz CSV fajla:")
print()
print(df.head())
print()
"""
Prvih 5 ucitanih kombinacija iz CSV fajla:

   0   1   2   3   4  5
0  8  10  15  21  31  6
1  7  16  19  25  31  2
2  1   7  18  25  28  7
3  6   7  12  19  22  2
4  8  25  29  33  34  3
"""

print()
print("Zadnjih 5 ucitanih kombinacija iz CSV fajla:")
print()
print(df.tail())
print()
"""
Zadnjih 5 ucitanih kombinacija iz CSV fajla:

    0   1   2   3   4  5
81  2   6  15  17  20  3
82  3   5  20  28  35  9
83  1   7  11  28  31  4
84  8  11  16  22  27  6
85  7   9  13  14  28  2
"""
####################################


# 2. Minimalni i maksimalni dozvoljeni brojevi po poziciji
min_val = [1, 2, 3, 4, 5, 1]
max_val = [31, 32, 33, 34, 35, 10]

# 3. Funkcija za mapiranje brojeva u indeksirani opseg [0..range_size-1]
def map_to_indexed_range(df, min_val, max_val):
    df_indexed = df.copy()
    for i in range(df.shape[1]):
        df_indexed[i] = df[i] - min_val[i]
        # Provera da li su svi brojevi u validnom opsegu
        if not df_indexed[i].between(0, max_val[i] - min_val[i]).all():
            raise ValueError(f"Vrednosti u koloni {i} nisu u opsegu 0 do {max_val[i] - min_val[i]}")
    return df_indexed

# 4. Primeni mapiranje
df_indexed = map_to_indexed_range(df, min_val, max_val)

# 5. Provera rezultata
print()
print(f"Učitano kombinacija: {df.shape[0]}, Broj pozicija: {df.shape[1]}")
print()
"""
Učitano kombinacija: 86, Broj pozicija: 6
"""


print()
print("Prvih 5 mapiranih kombinacija:")
print()
print(df_indexed.head())
print()
"""
Prvih 5 mapiranih kombinacija:

   0   1   2   3   4  5
0  7   8  12  17  26  5
1  6  14  16  21  26  1
2  0   5  15  21  23  6
3  5   5   9  15  17  1
4  7  23  26  29  29  2
"""

print()
print("Zadnjih 5 mapiranih kombinacija:")
print()
print(df_indexed.tail())
print()
"""
Zadnjih 5 mapiranih kombinacija:

    0  1   2   3   4  5
81  1  4  12  13  15  2
82  2  3  17  24  30  8
83  0  5   8  24  26  3
84  7  9  13  18  22  5
85  6  7  10  10  23  1
"""




# Parametri
num_qubits = 5          # 5 qubita po poziciji
num_layers = 2          # Dubina varijacionog sloja
num_positions = 5       # 6 pozicija (brojeva) u loto kombinaciji

# Enkoder: binarno kodiranje vrednosti u kvantni registar
def encode_position(value):
    """
    Sigurno enkoduje 'value' u QuantumCircuit sa tacno num_qubits qubita.
    Ako value zahteva vise bitova od num_qubits, koristi se LSB (zadnjih num_qubits bitova),
    i ispisuje se upozorenje.
    """
    # osiguraj int
    v = int(value)
    bin_full = format(v, 'b')  # pravi binarni bez vodećih nula
    if len(bin_full) > num_qubits:
        # upozorenje: vrednost ne staje u broj qubita; koristimo zadnjih num_qubits bita (LSB)
        print(f"Upozorenje: value={v} zahteva {len(bin_full)} bitova, a num_qubits={num_qubits}. Koristim zadnjih {num_qubits} bita.")
        bin_repr = bin_full[-num_qubits:]
    else:
        bin_repr = bin_full.zfill(num_qubits)

    qc = QuantumCircuit(num_qubits)
    # reversed da bi LSB išao na qubit 0 (ako želiš suprotno, ukloni reversed)
    for i, bit in enumerate(reversed(bin_repr)):
        if bit == '1':
            qc.x(i)
    return qc



# Varijacioni sloj: Ry rotacije + CNOT lanac
def variational_layer(params):
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.ry(params[i], i)
    for i in range(num_qubits - 1):
        qc.cx(i, i+1)
    return qc

# QCBM ansambl: slojevi varijacionih blokova
def qcbm_ansatz(params):
    qc = QuantumCircuit(num_qubits)
    for layer in range(num_layers):
        start = layer * num_qubits
        end = (layer + 1) * num_qubits
        qc.compose(variational_layer(params[start:end]), inplace=True)
    return qc

# Kompletan QCBM za svih 7 pozicija
def full_qcbm(params_list, values):
    total_qubits = num_qubits * num_positions
    qc = QuantumCircuit(total_qubits)

    for pos in range(num_positions):
        start_q = pos * num_qubits
        end_q = start_q + num_qubits

        # Enkoduj vrednost za poziciju
        qc_enc = encode_position(values[pos])
        qc.compose(qc_enc, qubits=range(start_q, end_q), inplace=True)

        # Dodaj varijacioni ansambl
        qc_var = qcbm_ansatz(params_list[pos])
        qc.compose(qc_var, qubits=range(start_q, end_q), inplace=True)

    # Dodaj merenja za svih 30 qubita
    qc.measure_all()

    return qc

# Test primer: enkoduj kombinaciju [13, 5, 7, 20, 23, 8]
test_values = [27,16,35,34,12,4]
np.random.seed(35)
params_list = [np.random.uniform(0, 2*np.pi, num_layers * num_qubits) for _ in range(num_positions)]

# Generiši QCBM za svih 6 pozicija
full_circuit = full_qcbm(params_list, test_values)



# Prikaz celog kruga u 'mpl' formatu
full_circuit.draw('mpl')
# plt.show()

# fold=40 prelama linije tako da veliki krug stane na ekran.
full_circuit.draw('mpl', fold=40)
# plt.show()


# The only valid choices are 
# text, latex, latex_source, and mpl


# Kompaktni prikaz kola
print("\nKompaktni prikaz kvantnog kola (text):\n")
# print(full_circuit.draw('text'))
"""
Kompaktni prikaz kvantnog kola (text):


"""


# display(full_circuit.draw())     
display(full_circuit.draw("mpl"))
# plt.show()


circuit_drawer(full_circuit, output='latex', style={"backgroundcolor": "#EEEEEE"})
# plt.show()


# import tinytex
# pip install tinycio
# pip install torchvision
# tinytex.install()



"""
# Sačuvaj kao PDF
img1 = full_circuit.draw('latex')
img1.save("/Users/milan/Desktop/GHQ/data/qc30_5_1.pdf")


# Sačuvaj kao sliku u latex formatu jpg
img2 = full_circuit.draw('latex')
img2.save("/Users/milan/Desktop/GHQ/data/qc30_5_2.jpg")


# Sačuvaj kao sliku u latex formatu png
img3 = full_circuit.draw('latex')
img3.save("/Users/milan/Desktop/GHQ/data/qc30_5_3.png")


# Sačuvaj kao sliku u matplotlib formatu jpg
img4 = full_circuit.draw('mpl', fold=40)
img4.savefig("/Users/milan/Desktop/GHQ/data/qc30_5_4.jpg")

# Sačuvaj kao sliku u matplotlib formatu png
img5 = full_circuit.draw('mpl', fold=40)
img5.savefig("/Users/milan/Desktop/GHQ/data/qc30_5_5.png")
"""




# Sačuvaj kao sliku u matplotlib formatu jpg
img4 = full_circuit.draw('mpl', fold=40)
img4.savefig("/Users/milan/Desktop/GHQ/KvantniRegresor/3QCBM/QCBM_qc25_5_4.jpg")



###############################################



# QCBM (Quantum Circuit Born Machine)

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from qiskit_aer import AerSimulator
from qiskit.circuit.library import TwoLocal
from qiskit_aer.primitives import Sampler
from qiskit_machine_learning.optimizers import COBYLA

from qiskit_machine_learning.utils import algorithm_globals
from tqdm import tqdm
import random

from qiskit import QuantumCircuit






# =========================
# 2. Koristimo samo zadnjih N=1000 za test
# =========================
N = 89
df = df.tail(N).reset_index(drop=True)






X = df.iloc[:, :-1].values  # prvih 5 brojeva
y_full = df.values          # svi 6 brojeva (5+1)

# Skaliranje X
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X).astype(np.float64)



print()
print("X_scaled.shape[0]")
print(X_scaled.shape[0])
print()
"""
X_scaled.shape[0]
87
"""

print()
print("len(X_scaled)")
print(len(X_scaled))
print()
"""
len(X_scaled)
87
"""


# =========================
# QCBM trening i predikcija
# =========================
predicted_combination = []

sampler = Sampler()
backend = AerSimulator()
# num_qubits = X_scaled.shape[1] * 3  
# malo veća reprezentacija prostora

# kreiranje QCBM ansatza
ansatz = TwoLocal(num_qubits=num_qubits,
                  rotation_blocks='ry',
                  entanglement_blocks='cz',
                  entanglement='full',
                  reps=2)

params = np.random.uniform(0, 2*np.pi, ansatz.num_parameters)


optimizer = COBYLA(maxiter=len(X_scaled))


# priprema target distribucije
def numbers_to_bitstring(row, n_qubits):
    # mapira sve brojeve u jedan bitstring dužine n_qubits
    return ''.join([format(int(v)-1, '06b') for v in row])

target_counts = {}
for row in y_full:
    bitstr = numbers_to_bitstring(row, num_qubits)
    target_counts[bitstr] = target_counts.get(bitstr, 0) + 1
for k in target_counts:
    target_counts[k] /= len(y_full)

# cost funkcija
def cost(theta):
    circ = ansatz.assign_parameters(theta)
    job = sampler.run([circ])
    result = job.result()
    counts = result.quasi_dists[0]
    loss = 0
    for bitstr, p in target_counts.items():
        q = counts.get(int(bitstr, 2), 1e-9)
        loss += p * np.log(p / q)
    return loss

# trening sa progress barom
pbar = tqdm(total=len(X_scaled), desc="QCBM trening")
# pbar = tqdm(total=num_samples, desc=f"Broj {i+1}")

theta = params.copy()
for _ in range(len(X_scaled)):
    # theta = optimizer(lambda th: cost(th), theta)
    pbar.update(1)
pbar.close()



# generisanje uzoraka
# final_circ = QuantumCircuit(num_qubits, num_qubits)  
# dodaj klasične bitove

full_circuit.compose(ansatz)
# final_circ.measure(range(num_qubits), range(num_qubits))  # merenja

# full_circuit.compose(ansatz.assign_parameters(theta), inplace=True)
# full_circuit.measure(range(num_qubits), range(num_qubits))  # merenja




#job = sampler.run([final_circ], shots=5000)
job = sampler.run([full_circuit], shots=100)

result = job.result()
counts = result.quasi_dists[0]




# sortiraj po verovatnoći
sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)


def bitstring_to_loto(bitstring_int):
    bitstring = format(bitstring_int, 'b').zfill(34)  # popuni do 34 bita
    main_numbers = [int(bitstring[i:i+6], 2) % 35 + 1 for i in range(0, 30, 6)]
    bonus_number = int(bitstring[30:34], 2) % 10 + 1
    return main_numbers + [bonus_number]




# Pretpostavljam da si već generisao counts iz Sampler-a i uzeo najverovatniji bitstring
most_prob_bitstring = max(counts, key=counts.get)
predicted_combo = bitstring_to_loto(most_prob_bitstring)
# predicted_combination.append(predicted_combo)

print()
print("\n=== Predviđena sledeća kombinacija (5+1) ===")
print("Kombinacija:", predicted_combo, f"(p={counts[most_prob_bitstring]:.4f})")
print()

print()
print("Broj različitih kombinacija u distribuciji:", len(counts))
print()
print("Top 5 najverovatnijih kombinacija:")
for bitstr, prob in sorted_counts[:5]:
    combo = bitstring_to_loto(bitstr)
    print(f"Kombinacija: {combo}, Verovatnoća: {prob:.4f}")
print()
"""
89
89
=== Predviđena sledeća kombinacija (5+1) ===
Kombinacija: [1, 4, 4, 10, 14, 10] (p=0.0200)
"""
