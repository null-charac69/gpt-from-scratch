"""
Byte-Pair Encoding (BPE) Tokenizer implementation.
This is similar to the tokenizer used in GPT-2.
"""

import regex as re
import json
import os
from typing import List, Dict


class SimpleTokenizer:
    """
    A simple character-level tokenizer for basic demonstrations.
    For production, you'd want to use GPT-2 BPE tokenizer.
    """
    
    def __init__(self, chars: str = None):
        if chars is None:
            # Default character set (printable ASCII + common chars)
            chars = ""
            for i in range(32, 127):
                chars += chr(i)
            chars += '\n\t'
        
        self.chars = sorted(list(set(chars)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
    
    def encode(self, text: str) -> List[int]:
        """Convert text to list of token ids."""
        return [self.char_to_idx.get(ch, 0) for ch in text]
    
    def decode(self, tokens: List[int]) -> str:
        """Convert list of token ids back to text."""
        return ''.join([self.idx_to_char.get(i, '') for i in tokens])
    
    def build_from_text(self, text: str):
        """Build vocabulary from text."""
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}


class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer.
    Implements the same tokenization scheme as GPT-2.
    """
    
    def __init__(self, vocab_file=None, merges_file=None):
        """
        Initialize BPE tokenizer.
        
        Args:
            vocab_file: Path to vocabulary JSON file
            merges_file: Path to merges text file
        """
        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
        if vocab_file and merges_file and os.path.exists(vocab_file):
            self._load_vocab(vocab_file, merges_file)
        else:
            # Initialize with minimal vocab
            self.encoder = {}
            self.decoder = {}
            self.bpe_ranks = {}
            self.vocab_size = 0
    
    def _bytes_to_unicode(self) -> Dict[int, str]:
        """
        Returns list of utf-8 byte and a mapping to unicode strings.
        Avoids mapping to whitespace/control characters.
        """
        bs = list(range(ord("!"), ord("~") + 1)) + \
             list(range(ord("¡"), ord("¬") + 1)) + \
             list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))
    
    def _get_pairs(self, word):
        """Return set of symbol pairs in a word."""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    def _bpe(self, token: str) -> str:
        """
        Apply BPE merges to a token.
        """
        if not self.bpe_ranks:
            return token
        
        word = tuple(token)
        pairs = self._get_pairs(word)
        
        if not pairs:
            return token
        
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = self._get_pairs(word)
        
        return ' '.join(word)
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token ids.
        """
        # Pattern to split on
        pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
        bpe_tokens = []
        for token in re.findall(pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder.get(bpe_token, 0) 
                            for bpe_token in self._bpe(token).split(' '))
        return bpe_tokens
    
    def decode(self, tokens: List[int]) -> str:
        """
        Decode token ids back to text.
        """
        text = ''.join([self.decoder.get(token, '') for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors='ignore')
        return text
    
    def _load_vocab(self, vocab_file: str, merges_file: str):
        """Load vocabulary and merges from files."""
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}
        
        with open(merges_file, 'r', encoding='utf-8') as f:
            bpe_data = f.read().split('\n')[1:-1]  # Skip header and empty last line
        
        bpe_merges = [tuple(merge.split()) for merge in bpe_data]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.vocab_size = len(self.encoder)
    
    def train(self, text: str, vocab_size: int = 5000, min_frequency: int = 2):
        """
        Train BPE tokenizer on text.
        This is a simplified training procedure.
        
        Args:
            text: Training text
            vocab_size: Target vocabulary size
            min_frequency: Minimum frequency for a merge
        """
        # Get byte-level tokens
        tokens = [self.byte_encoder[b] for b in text.encode('utf-8')]
        
        # Initialize vocabulary with individual characters
        vocab = {char: i for i, char in enumerate(sorted(set(tokens)))}
        
        # Iteratively merge most frequent pairs
        merges = []
        current_tokens = tokens[:]
        
        while len(vocab) < vocab_size:
            # Count pair frequencies
            pairs = {}
            for i in range(len(current_tokens) - 1):
                pair = (current_tokens[i], current_tokens[i + 1])
                pairs[pair] = pairs.get(pair, 0) + 1
            
            if not pairs:
                break
            
            # Find most frequent pair
            best_pair = max(pairs.items(), key=lambda x: x[1])
            if best_pair[1] < min_frequency:
                break
            
            # Merge the pair
            pair = best_pair[0]
            merges.append(pair)
            new_token = pair[0] + pair[1]
            vocab[new_token] = len(vocab)
            
            # Replace pair in tokens
            new_tokens = []
            i = 0
            while i < len(current_tokens):
                if i < len(current_tokens) - 1 and \
                   current_tokens[i] == pair[0] and current_tokens[i + 1] == pair[1]:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(current_tokens[i])
                    i += 1
            current_tokens = new_tokens
        
        self.encoder = vocab
        self.decoder = {v: k for k, v in vocab.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.vocab_size = len(vocab)