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
        """Devuelve una representación jerárquica del árbol."""
        result = "   " * level + f"{prefix}({self.char}:{self.data}):{cumchain}\n"
        if self.left:
            result += self.left.__str__(level + 1, prefix="L-0- ", cumchain=cumchain+'0')
        if self.right:
            result += self.right.__str__(level + 1, prefix="R-1- ", cumchain=cumchain+'1')
        return result
    

# Function to traverse tree in preorder 
# manner and push the huffman representation 
# of each character.
def preOrder(root, ans, curr):
    if root is None:
        return

    # Leaf node represents a character.
    if root.left is None and root.right is None:
        ans[root.char] = curr
        return

    preOrder(root.left, ans, curr + '0')
    preOrder(root.right, ans, curr + '1')


def huffmanCodes(s, freq):
    # Code here
    n = len(s)

    # Min heap for node class.
    pq = []
    for i in range(n):
        tmp = Node(freq[i], s[i])
        heapq.heappush(pq, tmp)

    # Construct huffman tree.
    while len(pq) >= 2:
        # Left node 
        l = heapq.heappop(pq)

        # Right node 
        r = heapq.heappop(pq)

        newNode = Node(l.data + r.data)
        newNode.left = l
        newNode.right = r

        heapq.heappush(pq, newNode)

    root = heapq.heappop(pq)
    codes_dic = {}
    preOrder(root, codes_dic, "")
    return codes_dic, root



# FRECUENCIAS Y PROMEDIO DE BITS


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
    total_symbols = sum(freq_counter.values())
    if total_symbols == 0:
        return 0.0

    expected_len = 0.0
    for c, freq in freq_counter.items():
        p_c = freq / total_symbols
        code_len = len(codes_dic[c])
        expected_len += p_c * code_len
    
    return expected_len


# COMPRESIÓN A .huff

def compress_file(input_path, output_path):
    import pickle
    from bitarray import bitarray

    # Leer archivo y construir frecuencias
    chars, freqs, counter, text = compute_frequencies_from_file(input_path)

    # Construir árbol de Huffman y códigos (string)
    codes_dic, root = huffmanCodes(chars, freqs)

    # Construir diccionario de códigos en bitarray
    codes_bits = {}
    for c, code_str in codes_dic.items():
        codes_bits[c] = bitarray(code_str)

    # Construir el flujo de bits comprimidos
    compressed_bits = bitarray()
    for ch in text:
        compressed_bits.extend(codes_bits[ch])

    # Calcular padding (relleno para completar el último byte)
    padding = (8 - (len(compressed_bits) % 8)) % 8
    if padding > 0:
        compressed_bits.extend('0' * padding)
        
    # Header que se guardará con pickle
    header = {
        'codes_dic': codes_dic,  # códigos como strings
        'padding': padding,      # número de bits de relleno al final
        'root': root             # árbol de Huffman para decodificación por árbol
    }

    # Escribir archivo .huff: primero header (pickle), luego bits comprimidos
    with open(output_path, 'wb') as f:
        pickle.dump(header, f)
        f.write(compressed_bits.tobytes())

    # Info útil en consola
    original_size_bytes = Path(input_path).stat().st_size
    compressed_size_bytes = Path(output_path).stat().st_size

    print(f"Archivo original : {input_path}")
    print(f"Archivo .huff     : {output_path}")
    print(f"Tamaño original   : {original_size_bytes} bytes")
    print(f"Tamaño comprimido : {compressed_size_bytes} bytes")
    print(f"Padding (bits)    : {padding}")


# DESCOMPRESIÓN USANDO DICCIONARIO INVERSO

def decompress_file_with_dict(input_path, output_path):
    import pickle
    from bitarray import bitarray

    # Leer header y bytes comprimidos
    with open(input_path, 'rb') as f:
        header = pickle.load(f)
        compressed_bytes = f.read()

    codes_dic = header['codes_dic']
    padding = header['padding']

    # Reconstruir bitarray completo
    bits = bitarray()
    bits.frombytes(compressed_bytes)

    # Quitar padding
    if padding > 0:
        bits = bits[:-padding]

    # Diccionario inverso: código (str) -> carácter
    inverse_codes = {code_str: ch for ch, code_str in codes_dic.items()}

    decoded_chars = []
    current_code = ""

    for bit in bits.to01():
        current_code += bit
        if current_code in inverse_codes:
            decoded_chars.append(inverse_codes[current_code])
            current_code = ""

    if current_code != "":
        print("⚠️ Advertencia: quedaron bits sin decodificar:", current_code)

    text = ''.join(decoded_chars)
    Path(output_path).write_text(text, encoding="utf-8")

    compressed_size_bytes = Path(input_path).stat().st_size
    restored_size_bytes = Path(output_path).stat().st_size

    print(f"Archivo .huff        : {input_path}")
    print(f"Archivo reconstruido : {output_path}")
    print(f"Tamaño .huff         : {compressed_size_bytes} bytes")
    print(f"Tamaño reconstruido  : {restored_size_bytes} bytes")


# DESCOMPRESIÓN USANDO EL ÁRBOL DE HUFFMAN

def decompress_file_with_tree(input_path, output_path):
    import pickle
    from bitarray import bitarray

    # Leer header y bytes comprimidos
    with open(input_path, 'rb') as f:
        header = pickle.load(f)
        compressed_bytes = f.read()

    padding = header['padding']
    root = header['root']

    # Reconstruir bitarray completo
    bits = bitarray()
    bits.frombytes(compressed_bytes)

    # Quitar padding
    if padding > 0:
        bits = bits[:-padding]

    decoded_chars = []
    node = root

    for bit in bits.to01():
        if bit == '0':
            node = node.left
        else:
            node = node.right

        # Si es hoja
        if node.left is None and node.right is None:
            decoded_chars.append(node.char)
            node = root

    text = ''.join(decoded_chars)
    Path(output_path).write_text(text, encoding="utf-8")

    compressed_size_bytes = Path(input_path).stat().st_size
    restored_size_bytes = Path(output_path).stat().st_size

    print(f"Archivo .huff        : {input_path}")
    print(f"Archivo reconstruido : {output_path}")
    print(f"Tamaño .huff         : {compressed_size_bytes} bytes")
    print(f"Tamaño reconstruido  : {restored_size_bytes} bytes")


# VISUALIZACIÓN DEL ÁRBOL CON GRAPHVIZ

def visualize_huffman_tree(root, output_path="huffman_tree"):
    from graphviz import Digraph

    dot = Digraph(comment="Huffman Tree")

    def add_nodes_edges(node, node_id):
        if node.char:
            label = f"{repr(node.char)}\\n{node.data}"
        else:
            label = str(node.data)

        dot.node(node_id, label)

        if node.left:
            left_id = node_id + "0"
            add_nodes_edges(node.left, left_id)
            dot.edge(node_id, left_id, label="0")

        if node.right:
            right_id = node_id + "1"
            add_nodes_edges(node.right, right_id)
            dot.edge(node_id, right_id, label="1")

    add_nodes_edges(root, "R")
    output_file = dot.render(output_path, format="png", cleanup=True)
    print(f"✅ Árbol de Huffman guardado en: {output_file}")



# MAIN

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        print("Modo ejemplo (sin archivo de texto):")
        s = ["a","b","c","d","e","f"]
        freq = [5, 9, 12, 13, 16, 45]
        codes_dic, root = huffmanCodes(s, freq)
        print(root)

        codes_dic = dict(sorted(codes_dic.items()))

        for c in codes_dic:
            print(f"{c}={codes_dic[c]}")

    elif sys.argv[1] == "compress":
        # Uso: python 05_uffman.py compress input.txt [output.huff]
        if len(sys.argv) < 3:
            print("Uso: python 05_uffman.py compress input.txt [output.huff]")
            sys.exit(1)

        input_path = sys.argv[2]
        if len(sys.argv) >= 4:
            output_path = sys.argv[3]
        else:
            # mismo nombre pero extensión .huff
            output_path = str(Path(input_path).with_suffix(".huff"))

        compress_file(input_path, output_path)

    elif sys.argv[1] == "decompress_dict":
        # Uso: python 05_uffman.py decompress_dict input.huff salida.txt
        if len(sys.argv) < 4:
            print("Uso: python 05_uffman.py decompress_dict input.huff salida.txt")
            sys.exit(1)

        input_huff = sys.argv[2]
        output_txt = sys.argv[3]
        decompress_file_with_dict(input_huff, output_txt)

    elif sys.argv[1] == "decompress_tree":
        # Uso: python 05_uffman.py decompress_tree input.huff salida.txt
        if len(sys.argv) < 4:
            print("Uso: python 05_uffman.py decompress_tree input.huff salida.txt")
            sys.exit(1)

        input_huff = sys.argv[2]
        output_txt = sys.argv[3]
        decompress_file_with_tree(input_huff, output_txt)

    else:
        # Modo análisis de archivo: python 05_uffman.py archivo.txt
        filename = sys.argv[1]
        print(f"Procesando archivo: {filename}")

        chars, freqs, counter, text = compute_frequencies_from_file(filename)

        print(f"Número total de símbolos (caracteres): {len(text)}")
        print(f"Número de caracteres distintos: {len(chars)}")

        codes_dic, root = huffmanCodes(chars, freqs)

        codes_dic_sorted = dict(sorted(codes_dic.items(), key=lambda x: x[0]))

        print("\nÁrbol de Huffman:")
        print(root)

        print("Códigos de Huffman por carácter:")
        for c, code in codes_dic_sorted.items():
            print(f"{repr(c)} -> {code}")

        avg_bits = average_bits_per_symbol(codes_dic, counter)
        print(f"\nNúmero promedio de bits por símbolo: {avg_bits:.4f}")

        total_symbols = sum(counter.values())
        total_bits = 0
        for c, freq in counter.items():
            total_bits += freq * len(codes_dic[c])

        print("\n=== Verificación manual ===")
        print(f"Total de bits generados: {total_bits}")
        print(f"Promedio bits/símbolo (manual) = {total_bits / total_symbols:.4f}")

        output_name = f"huffman_{Path(filename).stem}"
        visualize_huffman_tree(root, output_path=output_name)
