"""
===============================================================
 Program : 05_huffman.py
 Author  : Juan Manuel Ahuactzin
 Date    : 2025-10-13
 Version : 1.0
===============================================================
Description:
    Implementation of Huffman coding using a min-heap-based 
    approach to build the optimal binary prefix tree.
    Each node represents a character and its frequency, and 
    the resulting tree provides the Huffman code for each symbol.
    The class includes a __str__ method to visualize the tree
    structure and binary code paths (0/1).

	Adapted from: GeeksforGeeks. (s. f.). Huffman Coding | Greedy 
	Algo-3. GeeksforGeeks. 
	https://www.geeksforgeeks.org/dsa/huffman-coding-greedy-algo-3/


Notes:
    - The program constructs the Huffman tree and prints its
      hierarchical representation, followed by the Huffman code
      assigned to each character.
    - The __str__ method allows visualization of the recursive 
      structure, indicating '0' for left edges and '1' for right 
      edges.
===============================================================
"""

import heapq
from collections import Counter
from pathlib import Path

# Class to represent huffman tree
class Node:
    def __init__(self, x, character=""):
        self.data = x
        self.char = character
        self.left = None
        self.right = None
        
    def __lt__(self, other):
        return self.data < other.data
    
    def __str__(self, level=0, prefix="Root: ", cumchain=""):
        result = "   " * level + f"{prefix}({self.char}:{self.data}):{cumchain}\n"
        if self.left:
            result += self.left.__str__(level + 1, prefix="L-0- ", cumchain=cumchain+'0')
        if self.right:
            result += self.right.__str__(level + 1, prefix="R-1- ", cumchain=cumchain+'1')
        return result



# Function to traverse tree in preorder manner and push the huffman representation of each characte
def preOrder(root, ans, curr):
    if root is None:
        return
    if root.left is None and root.right is None:
        ans[root.char] = curr
        return
    preOrder(root.left, ans, curr + '0')
    preOrder(root.right, ans, curr + '1')


def huffmanCodes(s, freq):
    pq = []
    for i in range(len(s)):
        heapq.heappush(pq, Node(freq[i], s[i]))

    while len(pq) >= 2:
        l = heapq.heappop(pq)
        r = heapq.heappop(pq)
        newNode = Node(l.data + r.data)
        newNode.left = l
        newNode.right = r
        heapq.heappush(pq, newNode)

    root = heapq.heappop(pq)
    codes = {}
    preOrder(root, codes, "")
    return codes, root

#   FRECUENCIAS Y PROMEDIO
def compute_frequencies_from_text(text):
    counter = Counter(text)
    chars = list(counter.keys())
    freqs = [counter[c] for c in chars]
    return chars, freqs, counter


def compute_frequencies_from_file(path):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    chars, freqs, counter = compute_frequencies_from_text(text)
    return chars, freqs, counter, text


def average_bits_per_symbol(codes_dic, freq_counter):
    total = sum(freq_counter.values())
    if total == 0:
        return 0.0

    E = 0.0
    for c, f in freq_counter.items():
        E += (f / total) * len(codes_dic[c])
    return E


#   COMPRESIÓN

def compress_file(input_path, output_path):
    import pickle
    from bitarray import bitarray

    chars, freqs, counter, text = compute_frequencies_from_file(input_path)
    codes_dic, root = huffmanCodes(chars, freqs)

    codes_bits = {c: bitarray(code) for c, code in codes_dic.items()}

    compressed_bits = bitarray()
    for ch in text:
        compressed_bits.extend(codes_bits[ch])

    padding = (8 - (len(compressed_bits) % 8)) % 8
    if padding > 0:
        compressed_bits.extend("0" * padding)

    header = {
        "codes_dic": codes_dic,
        "padding": padding,
        "root": root
    }

    with open(output_path, "wb") as f:
        pickle.dump(header, f)
        f.write(compressed_bits.tobytes())

    orig = Path(input_path).stat().st_size
    comp = Path(output_path).stat().st_size

    print(f"Archivo original   : {orig} bytes")
    print(f"Archivo comprimido : {comp} bytes")
    print(f"Padding            : {padding} bits")


#   DESCOMPRESIÓN (diccionario inverso)

def decompress_file_with_dict(input_path, output_path):
    import pickle
    from bitarray import bitarray

    with open(input_path, "rb") as f:
        header = pickle.load(f)
        compressed_bytes = f.read()

    codes_dic = header["codes_dic"]
    padding = header["padding"]

    bits = bitarray()
    bits.frombytes(compressed_bytes)
    if padding > 0:
        bits = bits[:-padding]

    inverse = {code: char for char, code in codes_dic.items()}

    decoded = []
    current = ""

    for b in bits.to01():
        current += b
        if current in inverse:
            decoded.append(inverse[current])
            current = ""

    Path(output_path).write_text("".join(decoded), encoding="utf-8")
    print(f"Archivo decodificado → {output_path}")


#   DESCOMPRESIÓN (árbol)

def decompress_file_with_tree(input_path, output_path):
    import pickle
    from bitarray import bitarray

    with open(input_path, "rb") as f:
        header = pickle.load(f)
        compressed_bytes = f.read()

    root = header["root"]
    padding = header["padding"]

    bits = bitarray()
    bits.frombytes(compressed_bytes)
    if padding > 0:
        bits = bits[:-padding]

    decoded = []
    node = root

    for b in bits.to01():
        node = node.left if b == "0" else node.right
        if node.left is None and node.right is None:
            decoded.append(node.char)
            node = root

    Path(output_path).write_text("".join(decoded), encoding="utf-8")
    print(f"Archivo decodificado → {output_path}")


#   VISUALIZACIÓN (Graphviz)

def visualize_huffman_tree(root, output_path="huffman_tree"):
    from graphviz import Digraph

    dot = Digraph()

    def add(node, node_id):
        label = f"{repr(node.char)}\\n{node.data}" if node.char else str(node.data)
        dot.node(node_id, label)
        if node.left:
            add(node.left, node_id + "0")
            dot.edge(node_id, node_id + "0", label="0")
        if node.right:
            add(node.right, node_id + "1")
            dot.edge(node_id, node_id + "1", label="1")

    add(root, "R")
    out = dot.render(output_path, format="png", cleanup=True)
    print("Árbol guardado en:", out)


#   ESTADÍSTICAS 30 ARCHIVOS

def run_batch_statistics(base="datasets"):
    import csv

    rows = []

    for lang in ["es", "fr", "en"]:
        for file in sorted((Path(base) / lang).glob("*.txt")):

            text = file.read_text(encoding="utf-8")
            chars, freqs, counter = compute_frequencies_from_text(text)
            codes_dic, root = huffmanCodes(chars, freqs)

            avg_bits = average_bits_per_symbol(codes_dic, counter)

            output_huff = file.with_suffix(".huff")
            compress_file(str(file), str(output_huff))

            orig = file.stat().st_size
            comp = output_huff.stat().st_size
            factor = comp / orig if orig > 0 else 0

            rows.append([file.name, lang.upper(), orig, comp, factor, avg_bits])

    with open("tabla_compresion.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Archivo", "Idioma", "Original", "Comprimido", "Factor", "Bits/símbolo"])
        w.writerows(rows)

    print("tabla_compresion.csv generada")


#   RESUMEN POR IDIOMA

def summarize_by_language(input_csv="tabla_compresion.csv"):
    import csv
    from statistics import mean

    factors = {"ES": [], "FR": [], "EN": []}
    bits = {"ES": [], "FR": [], "EN": []}

    with open(input_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lang = row["Idioma"]
            factors[lang].append(float(row["Factor"]))
            bits[lang].append(float(row["Bits/símbolo"]))

    with open("resumen_idiomas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Idioma", "Promedio Factor", "Promedio Bits/símbolo"])
        for lang in ["ES", "FR", "EN"]:
            w.writerow([
                lang,
                mean(factors[lang]),
                mean(bits[lang])
            ])

    print("resumen_idiomas.csv generado")


#   BENCHMARK

def benchmark_decode(input_huff, lang, repeats=300):
    import time, pickle
    from statistics import mean, stdev
    from bitarray import bitarray

    with open(input_huff, "rb") as f:
        header = pickle.load(f)
        compressed_bytes = f.read()

    codes_dic = header["codes_dic"]
    padding = header["padding"]
    root = header["root"]

    bits = bitarray()
    bits.frombytes(compressed_bytes)
    if padding > 0:
        bits = bits[:-padding]
    bits_str = bits.to01()

    # diccionario
    inv = {code: ch for ch, code in codes_dic.items()}

    times_dict = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        decoded = []
        cur = ""
        for b in bits_str:
            cur += b
            if cur in inv:
                decoded.append(inv[cur])
                cur = ""
        times_dict.append(time.perf_counter() - t0)

    # árbol
    times_tree = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        decoded = []
        node = root
        for b in bits_str:
            node = node.left if b == "0" else node.right
            if node.left is None and node.right is None:
                decoded.append(node.char)
                node = root
        times_tree.append(time.perf_counter() - t0)

    print(f"\nBenchmark {lang}:")
    print("  Diccionario inverso → mean =", mean(times_dict), "std =", stdev(times_dict))
    print("  Árbol → mean =", mean(times_tree), "std =", stdev(times_tree))



#   MAIN

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        print("Modo ejemplo ejecutado.")

    elif sys.argv[1] == "compress":
        compress_file(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else Path(sys.argv[2]).with_suffix(".huff"))

    elif sys.argv[1] == "decompress_dict":
        decompress_file_with_dict(sys.argv[2], sys.argv[3])

    elif sys.argv[1] == "decompress_tree":
        decompress_file_with_tree(sys.argv[2], sys.argv[3])

    elif sys.argv[1] == "stats":
        run_batch_statistics()

    elif sys.argv[1] == "summary":
        summarize_by_language()

    elif sys.argv[1] == "benchmark":
        benchmark_decode(sys.argv[2], sys.argv[3])

    else:
        filename = sys.argv[1]
        chars, freqs, counter, text = compute_frequencies_from_file(filename)
        codes_dic, root = huffmanCodes(chars, freqs)

        print(root)
        visualize_huffman_tree(root, "huffman_tree_out")
