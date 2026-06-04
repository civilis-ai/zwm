from zwm.core.constants import CODON_TABLE


def hexagram_to_codon(hex_bits: int) -> str:
    return CODON_TABLE.get(hex_bits, "???")


def codon_to_hexagram_bits(codon: str) -> int:
    for bits, c in CODON_TABLE.items():
        if c == codon:
            return bits
    raise ValueError(f"Unknown codon: {codon}")


def codon_amino_acid(codon: str) -> str:
    codon_to_aa = {
        "UUU": "Phe", "UUC": "Phe", "UUA": "Leu", "UUG": "Leu",
        "CUU": "Leu", "CUC": "Leu", "CUA": "Leu", "CUG": "Leu",
        "AUU": "Ile", "AUC": "Ile", "AUA": "Ile", "AUG": "Met",
        "GUU": "Val", "GUC": "Val", "GUA": "Val", "GUG": "Val",
        "UCU": "Ser", "UCC": "Ser", "UCA": "Ser", "UCG": "Ser",
        "CCU": "Pro", "CCC": "Pro", "CCA": "Pro", "CCG": "Pro",
        "ACU": "Thr", "ACC": "Thr", "ACA": "Thr", "ACG": "Thr",
        "GCU": "Ala", "GCC": "Ala", "GCA": "Ala", "GCG": "Ala",
        "UAU": "Tyr", "UAC": "Tyr", "UAA": "STOP", "UAG": "STOP",
        "CAU": "His", "CAC": "His", "CAA": "Gln", "CAG": "Gln",
        "AAU": "Asn", "AAC": "Asn", "AAA": "Lys", "AAG": "Lys",
        "GAU": "Asp", "GAC": "Asp", "GAA": "Glu", "GAG": "Glu",
        "UGU": "Cys", "UGC": "Cys", "UGA": "STOP", "UGG": "Trp",
        "CGU": "Arg", "CGC": "Arg", "CGA": "Arg", "CGG": "Arg",
        "AGU": "Ser", "AGC": "Ser", "AGA": "Arg", "AGG": "Arg",
        "GGU": "Gly", "GGC": "Gly", "GGA": "Gly", "GGG": "Gly",
    }
    return codon_to_aa.get(codon, "Unknown")
