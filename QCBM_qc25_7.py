# QCBM (Quantum Circuit Born Machine)


"""
Loto Skraceni Sistemi 
https://www.lotoss.info
ABBREVIATED LOTTO SYSTEMS
"""


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
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED


"""
svih 4502 izvlacenja Loto 7/39 u Srbiji
30.07.1985.- 28.10.2025.
"""

# 1. Učitaj loto podatke
df = pd.read_csv("/data/loto7_4502_k85.csv", header=None)


###################################
print()
print("Prvih 5 ucitanih kombinacija iz CSV fajla:")
print()
print(df.head())
print()
"""
Prvih 5 ucitanih kombinacija iz CSV fajla:

"""

print()
print("Zadnjih 5 ucitanih kombinacija iz CSV fajla:")
print()
print(df.tail())
print()
"""
Zadnjih 5 ucitanih kombinacija iz CSV fajla:

"""
####################################


# 2. Minimalni i maksimalni dozvoljeni brojevi po poziciji
min_val = [1, 2, 3, 4, 5, 6, 7]
max_val = [33, 34, 35, 36, 37, 38, 39]

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
Učitano kombinacija: 4502, Broj pozicija: 7
"""


print()
print("Prvih 5 mapiranih kombinacija:")
print()
print(df_indexed.head())
print()
"""
Prvih 5 mapiranih kombinacija:


"""

print()
print("Zadnjih 5 mapiranih kombinacija:")
print()
print(df_indexed.tail())
print()
"""
Zadnjih 5 mapiranih kombinacija:


"""


import math

# Parametri
num_qubits = 5          # 5 qubita po poziciji
num_layers = 2          # Dubina varijacionog sloja
num_positions = 5       # 6 pozicija (brojeva) u loto kombinaciji

# Uzimamo broj pozicija direktno iz učitanog df (da bude konzistentno)
# num_positions = df.shape[1]  # npr. 6 ako CSV ima 6 kolona



# Enkoder: binarno kodiranje vrednosti u kvantni registar
# Funkcija koja enkodira vrednosti u qubitove
# Sad uzima u obzir ulazne kombinacije i proizvodi varijacije u zavisnosti od CSV
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




# Varijacioni sloj i ansatz ostaju, ali pazi da parametri poklapaju dimenzije:
def variational_layer(params):
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.ry(params[i], i)
    for i in range(num_qubits - 1):
        qc.cx(i, i+1)
    return qc




# QCBM ansatz ostaje isti, ali parametri se inicijalizuju u zavisnosti od ulaza
def qcbm_ansatz(params):
    qc = QuantumCircuit(num_qubits)
    for layer in range(num_layers):
        start = layer * num_qubits
        end = (layer + 1) * num_qubits
        qc.compose(variational_layer(params[start:end]), inplace=True)
    return qc





# Kompletan QCBM za svih 7 pozicija
# Kompletan QCBM gde ulaz iz CSV utiče na kolu
def full_qcbm(params_list, values):
    total_qubits = num_qubits * num_positions
    qc = QuantumCircuit(total_qubits)

    for pos in range(num_positions):
        start_q = pos * num_qubits
        end_q = start_q + num_qubits

        # Enkodiranje vrednosti za poziciju (ulaz iz CSV)
        qc_enc = encode_position(values[pos])
        qc.compose(qc_enc, qubits=range(start_q, end_q), inplace=True)

        # Varijacioni ansatz sa parametrima
        qc_var = qcbm_ansatz(params_list[pos])
        qc.compose(qc_var, qubits=range(start_q, end_q), inplace=True)

    qc.measure_all()
    return qc






np.random.seed(39)

params_list = [np.random.uniform(0, 2*np.pi, num_layers * num_qubits) for _ in range(num_positions)]




"""
if len(test_values) < num_positions:
    raise ValueError(f"test_values mora imati najmanje {num_positions} elemenata; trenutno ima {len(test_values)}.")
test_values_trim = test_values[:num_positions]
"""




# Prikaz celog kruga u 'mpl' formatu
# full_qcbm.draw('mpl')
# plt.show()

# fold=40 prelama linije tako da veliki krug stane na ekran.
# full_qcbm.draw('mpl', fold=40)
# plt.show()




# Kompaktni prikaz kola
print("\nKompaktni prikaz kvantnog kola (text):\n")
# print(full_circuit.draw('text'))
"""
Kompaktni prikaz kvantnog kola (text):


"""


# display(full_circuit.draw())     
# display(full_qcbm.draw("mpl"))
# plt.show()


# circuit_drawer(full_qcbm, output='latex', style={"backgroundcolor": "#EEEEEE"})
# plt.show()




"""
# Sačuvaj kao PDF
img1 = full_circuit.draw('latex')
img1.save("/data/qc25_5_1.pdf")


# Sačuvaj kao sliku u latex formatu jpg
img2 = full_circuit.draw('latex')
img2.save("/data/qc25_5_2.jpg")


# Sačuvaj kao sliku u latex formatu png
img3 = full_circuit.draw('latex')
img3.save("/data/qc25_5_3.png")


# Sačuvaj kao sliku u matplotlib formatu jpg
img4 = full_circuit.draw('mpl', fold=40)
img4.savefig("/data/qc25_5_4.jpg")

# Sačuvaj kao sliku u matplotlib formatu png
img5 = full_circuit.draw('mpl', fold=40)
img5.savefig("/data/qc25_5_5.png")



# Sačuvaj kao sliku u matplotlib formatu jpg
# img4 = full_qcbm.draw('mpl', fold=40)
# img4.savefig("/Users/milan/Desktop/GHQ/KvantniRegresor/3QCBM/QCBM_qc25_7_5.jpg")

"""

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

# Uzmi svih 450 izvlacenja iz CSV fajla
N = 4502 # možeš menjati N
df_tail = df.tail(N).reset_index(drop=True)
X_tail = df_tail.iloc[:, :-1].values  # prvih 5 brojeva

# Normalizacija X
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X_tail).astype(np.float64)

# Parametri QCBM zavise od srednjih vrednosti po koloni
params_list = []
for pos in range(num_positions):
    mean_val = int(np.round(X_scaled[:, pos].mean() * (2**num_qubits - 1)))
    layer_params = np.full(num_layers * num_qubits, mean_val / (2**num_qubits) * 2*np.pi)
    params_list.append(layer_params)









X = df.iloc[:, :-1].values  # prvih 5 brojeva
y_full = df.values          # svi 7 brojeva (5+2)




print()
print("X_scaled.shape[0]")
print(X_scaled.shape[0])
print()
"""
X_scaled.shape[0]
4502
"""

print()
print("len(X_scaled)")
print(len(X_scaled))
print()
"""
len(X_scaled)
4502
"""


# =========================
# QCBM trening i predikcija
# =========================
predicted_combination = []

sampler = Sampler()
backend = AerSimulator()


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





# Test vrednosti: uzmi srednju vrednost poslednjih N izvlačenja
test_values = [int(val) for val in X_tail.mean(axis=0)]

# Kreiraj QCBM kola
full_circuit = full_qcbm(params_list, test_values)

# Sampler
job = sampler.run([full_circuit], shots=1000000)
result = job.result()
counts = result.quasi_dists[0]





# sortiraj po verovatnoći
sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)


def bitstring_to_loto_with_7(bitstring_int, num_qubits=5, num_positions=5):
    """
    Mapira integer bitstring (iz Sampler-a) u num_positions brojeva,
    zatim deterministički računa 6. i 7. broj tako da budu u validnom opsegu
    i da se ne ponavljaju sa prethodnih num_positions brojeva.
    """
    num_bits = num_qubits * num_positions  # 25
    bitstring = format(int(bitstring_int), 'b').zfill(num_bits)

    main_numbers = []
    for pos in range(num_positions):
        start = pos * num_qubits
        chunk = bitstring[start:start + num_qubits]
        val = int(chunk, 2)
        mv = min_val[pos]
        Mv = max_val[pos]
        rng = Mv - mv + 1
        mapped = (val % rng) + mv
        main_numbers.append(int(mapped))

    # Helper: naći jedinstvenu vrednost u opsegu za poziciju idx, pokušavajući od start_val
    def find_unique(start_val, used_set, idx):
        mv = min_val[idx]; Mv = max_val[idx]; rng = Mv - mv + 1
        v = ((start_val - mv) % rng) + mv
        tries = 0
        while v in used_set and tries < rng:
            v = mv + ((v - mv + 1) % rng)
            tries += 1
        if v in used_set:
            # fallback: pronađi prvi slobodan
            for cand in range(mv, Mv + 1):
                if cand not in used_set:
                    v = cand
                    break
        return int(v)

    # Determinističko računjanje 6. i 7. broja (bez duplikata)
    sum_main = sum(main_numbers)
    # 6. broj koristi opseg index 5 (prethodni min_val/max_val def)
    start6 = (sum_main) % (max_val[5] - min_val[5] + 1) + min_val[5]
    sixth = find_unique(start6, set(main_numbers), 5)

    used = set(main_numbers) | {sixth}
    # 7. broj koristi opseg index 6 (min_val[6] je obavezno >= 7 ako je tako definisano)
    start7 = (sum_main + sixth) % (max_val[6] - min_val[6] + 1) + min_val[6]
    seventh = find_unique(start7, used, 6)

    return main_numbers + [sixth, seventh]



most_prob_bitstring = max(counts, key=counts.get)
predicted_combo = bitstring_to_loto_with_7(most_prob_bitstring,
                                           num_qubits=num_qubits,
                                           num_positions=num_positions)

print()
print("\n=== Predviđena sledeća kombinacija (7) ===")
print("Kombinacija:", predicted_combo, f"(p={counts[most_prob_bitstring]:.4f})")
print()
"""
4502
4502
=== Predviđena sledeća kombinacija (7) ===
Kombinacija: [12, 4, x, x, x, 38, 11], Verovatnoća: 0.0461
"""





print()
print("Broj različitih kombinacija u distribuciji:", len(counts))
print()
print("Top 5 najverovatnijih kombinacija:")
for bitstr, prob in sorted_counts[:5]:
    combo = bitstring_to_loto_with_7(bitstr)
    print(f"Kombinacija: {combo}, Verovatnoća: {prob:.4f}")
print()
"""

"""





"""
=== Qiskit Version Table ===
Software                       Version        
---------------------------------------------
qiskit                         1.4.4          
qiskit_machine_learning        0.8.3          

=== System Information ===
Python version                 3.11.13        
OS                             Darwin         
Time                           Tue Sep 09 18:11:49 2025 CEST
"""



