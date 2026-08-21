import gzip
import html
import os
import sys
from functools import lru_cache

import ftfy
import regex as re


# 在clip里的词表bpe_simple_vocab_16e6.txt.gz
@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")

"""
    是一个装饰器，用于缓存函数的返回值。这意味着如果函数被多次调用且输入相同，它将直接返回缓存的结果，而不是重新计算。
    这可以显著提高性能，尤其是在处理大量数据时。(类似哈希表)
"""
@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    """ftfy（Fixes Text For You）修复复文本中的常见问题。
        text = "This is a tÃªst."
        fixed_text = ftfy.fix_text(text)
        print(fixed_text)  # 输出: "This is a test."
    """
    text = ftfy.fix_text(text)
    """确保所有的 HTML 实体都被正确解码
        text = "&lt;p&gt;This is a &amp;lt;test&amp;gt;&lt;/p&gt;"
        unescaped_text = html.unescape(html.unescape(text))
        print(unescaped_text)  # 输出: "<p>This is a <test></p>"
    """
    text = html.unescape(html.unescape(text))
    return text.strip()     # 去除首尾空白字符


def whitespace_clean(text):

    """使用一个空格替换多个空格的问题
        text = "This   is  a\ttest\nstring."
        cleaned_text = re.sub(r'\s+', ' ', text)
        print(cleaned_text)  # 输出: "This is a test string."
    """
    text = re.sub(r'\s+', ' ', text)        # \s+：匹配一个或多个空白字符。空白字符包括空格、制表符、换行符等
    text = text.strip()                     # 去除首尾空白字符
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()      # 调用方法返回的字典{k (number), v (char)}
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}    # 改为{v (char)， k (number)}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')     # 解压一行一行读取，每行为列表中一个元素
        merges = merges[1:49152-256-2+1]        # 48895
        merges = [tuple(merge.split()) for merge in merges] # 按空格分割每行元素
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]   # '!' + '!</w>'
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))          # 词表大小: 49408
        self.decoder = {v: k for k, v in self.encoder.items()}      # 词表大小: 49408

        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
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
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    # token化
    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()      # 清理文本

        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text
