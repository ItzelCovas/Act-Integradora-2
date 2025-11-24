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
    """
    Construye el árbol de Huffman a partir de:
    - s: lista de caracteres
    - freq: lista de frecuencias (mismo orden que s)
    Devuelve:
    - codes_dic: diccionario {caracter: codigo_binario_str}
    - root: raíz del árbol de Huffman
    """
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
    """
    Recibe un string completo y devuelve:
    - lista de caracteres únicos (chars)
    - lista de frecuencias (freqs), alineada con chars
    - contador completo (Counter) por si lo queremos usar después
    """
    counter = Counter(text)
    # Quitamos posibles caracteres con frecuencia 0
    chars = list(counter.keys())
    freqs = [counter[c] for c in chars]
    return chars, freqs, counter


def compute_frequencies_from_file(path):
    """
    Lee un archivo de texto (UTF-8) y regresa:
    - chars, freqs, counter
    - el texto completo (por si lo queremos usar)
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    chars, freqs, counter = compute_frequencies_from_text(text)
    return chars, freqs, counter, text


def average_bits_per_symbol(codes_dic, freq_counter):
    """
    Calcula el número promedio de bits por símbolo:
    E[L] = sum( p(c) * len(code(c)) ) sobre todos los caracteres c
    donde p(c) = freq(c) / total_simbolos
    """
    total_symbols = sum(freq_counter.values())
    if total_symbols == 0:
        return 0.0

    expected_len = 0.0
    for c, freq in freq_counter.items():
        p_c = freq / total_symbols
        code_len = len(codes_dic[c])
        expected_len += p_c * code_len
    
    return expected_len


if __name__ == "__main__":
    import sys

    # Si no se pasa archivo, usamos el ejemplo de la clase
    if len(sys.argv) == 1:
        print("Modo ejemplo (sin archivo de texto):")
        s = ["a","b","c","d","e","f"]
        freq = [5, 9, 12, 13, 16, 45]
        codes_dic, root = huffmanCodes(s, freq)
        print(root)

        codes_dic = dict(sorted(codes_dic.items()))
        for c in codes_dic:
            print(f"{c} = {codes_dic[c]}")
    else:
        filename = sys.argv[1]
        print(f"Procesando archivo: {filename}")

        chars, freqs, counter, text = compute_frequencies_from_file(filename)

        print(f"Número total de símbolos (caracteres): {len(text)}")
        print(f"Número de caracteres distintos: {len(chars)}")

        codes_dic, root = huffmanCodes(chars, freqs)

        # Ordenamos el diccionario por carácter para verlo bonito
        codes_dic_sorted = dict(sorted(codes_dic.items(), key=lambda x: x[0]))

        print("\nÁrbol de Huffman:")
        print(root)

        print("Códigos de Huffman por carácter:")
        for c, code in codes_dic_sorted.items():
            # Mostramos repr(c) para ver espacios, saltos de línea, etc.
            print(f"{repr(c)} -> {code}")

        avg_bits = average_bits_per_symbol(codes_dic, counter)
        print(f"\nNúmero promedio de bits por símbolo: {avg_bits:.4f}")
        
        

