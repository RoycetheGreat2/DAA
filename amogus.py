import time
import random
import string
# Your Boyer-Moore variant
def bad_char_heuristic(pattern):
    bad_char = {}
    for i, char in enumerate(pattern):
        bad_char[char] = i
    return bad_char

def boyer_moore_optimized(text, pattern):
    m, n = len(pattern), len(text)
    if m == 0:
        return list(range(n + 1))
    if n == 0 or m > n:
        return []

    bad_char = bad_char_heuristic(pattern)
    positions = []
    s = 0

    while s <= n - m:
        left = 0
        right = m - 1
        mismatch_pos = -1

        while left <= right:
            if pattern[right] != text[s + right]:
                mismatch_pos = right
                break
            if pattern[left] != text[s + left]:
                mismatch_pos = left
                break
            left += 1
            right -= 1

        if mismatch_pos == -1:
            positions.append(s)
            s += 1
        else:
            mismatch_char = text[s + mismatch_pos]
            last_occurrence = bad_char.get(mismatch_char, -1)
            shift = max(1, mismatch_pos - last_occurrence)
            s += shift

    return positions

# Standard Boyer-Moore with bad character + good suffix
def preprocess_bad_char(pattern):
    bad_char = [-1] * 256
    for i, c in enumerate(pattern):
        bad_char[ord(c)] = i
    return bad_char

def preprocess_suffixes(pattern):
    m = len(pattern)
    suff = [0] * m
    suff[m-1] = m
    g = m - 1
    f = 0
    for i in range(m-2, -1, -1):
        if i > g and suff[i + m - 1 - f] < i - g:
            suff[i] = suff[i + m - 1 - f]
        else:
            if i < g:
                g = i
            f = i
            while g >= 0 and pattern[g] == pattern[g + m - 1 - f]:
                g -= 1
            suff[i] = f - g
    return suff

def preprocess_good_suffix(pattern):
    m = len(pattern)
    suff = preprocess_suffixes(pattern)
    good_suffix = [m] * m
    j = 0
    for i in range(m - 1, -1, -1):
        if suff[i] == i + 1:
            for j in range(m - 1 - i):
                if good_suffix[j] == m:
                    good_suffix[j] = m - 1 - i
    for i in range(m - 1):
        good_suffix[m - 1 - suff[i]] = m - 1 - i
    return good_suffix

def boyer_moore_standard(text, pattern):
    m = len(pattern)
    n = len(text)
    if m == 0:
        return list(range(n + 1))
    if n == 0 or m > n:
        return []

    bad_char = preprocess_bad_char(pattern)
    good_suffix = preprocess_good_suffix(pattern)

    positions = []
    s = 0
    while s <= n - m:
        j = m - 1
        while j >= 0 and pattern[j] == text[s + j]:
            j -= 1
        if j < 0:
            positions.append(s)
            s += good_suffix[0] if m > 1 else 1
        else:
            bc_shift = j - bad_char[ord(text[s + j])]
            gs_shift = good_suffix[j]
            s += max(1, max(bc_shift, gs_shift))
    return positions

# Built-in find all occurrences
def builtin_find_all(text, pattern):
    positions = []
    start = 0
    while True:
        pos = text.find(pattern, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1
    return positions

# KMP Implementation
def compute_lps(pattern):
    lps = [0] * len(pattern)
    length = 0
    i = 1
    while i < len(pattern):
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length-1]
            else:
                lps[i] = 0
                i += 1
    return lps

def kmp_search(text, pattern):
    if pattern == "":
        return list(range(len(text)+1))
    lps = compute_lps(pattern)
    positions = []
    i = j = 0
    while i < len(text):
        if pattern[j] == text[i]:
            i += 1
            j += 1
        if j == len(pattern):
            positions.append(i - j)
            j = lps[j-1]
        elif i < len(text) and pattern[j] != text[i]:
            if j != 0:
                j = lps[j-1]
            else:
                i += 1
    return positions

# Rabin-Karp Implementation
def rabin_karp_search(text, pattern, base=256, mod=10**9+7):
    m, n = len(pattern), len(text)
    if m == 0:
        return list(range(n+1))
    if n == 0 or m > n:
        return []
    
    hpattern = 0
    htext = 0
    h = 1
    positions = []

    for i in range(m-1):
        h = (h * base) % mod

    for i in range(m):
        hpattern = (base * hpattern + ord(pattern[i])) % mod
        htext = (base * htext + ord(text[i])) % mod

    for i in range(n - m + 1):
        if hpattern == htext and text[i:i+m] == pattern:
            positions.append(i)
        if i < n - m:
            htext = (base * (htext - ord(text[i]) * h) + ord(text[i+m])) % mod
            if htext < 0:
                htext += mod

    return positions

# Benchmarking helper
def benchmark_search(func, text, pattern, repeat=1000):
    start = time.perf_counter()
    for _ in range(repeat):
        func(text, pattern)
    end = time.perf_counter()
    return (end - start) * 1000 / repeat  # ms

# Test cases
def test_cases():
    cases = [
        ("aaaaaa", "aa"),                      # overlapping occurrences
        ("abracadabra", "abra"),               # pattern appears twice at start and end
        ("", ""),                             # both empty
        ("abcabcabc", "abc"),                 # repeating pattern
        ("abcabcabc", "abcd"),                # pattern longer than any occurrence
        ("the quick brown fox jumps over the lazy dog", "the"),  # normal sentence
        ("a" * 10000 + "b", "ab"),            # very large text, pattern near end
        ("mississippi", "issi"),              # pattern inside text with repeats
        ("abcdefg", ""),                      # empty pattern
        ("abcdefg", "hij"),                   # pattern not in text
        ("a", "a"),                          # single char text and pattern match
        ("a", "b"),                          # single char text and pattern no match
        ("abc", "abcd"),                     # pattern longer than text
        ("aaaaa", "aaa"),                    # overlapping patterns
        ("ababababab", "abab"),              # overlapping pattern occurrences
        ("xyz", "z"),                        # pattern at end of text
        ("xyz", "x"),                        # pattern at start of text
        ("test pattern test pattern", "pattern"),  # multiple occurrences
        ("1234567890", "789"),               # numeric text and pattern
        ("", "a"),                          # empty text, non-empty pattern
        ("a", ""),                         # non-empty text, empty pattern
        ("abcabcabcabcabcabcabcabcabcabc", "abcabcabc"),  # very repetitive
    ]
    random.seed(42)
    long_text = ''.join(random.choices(string.ascii_lowercase + string.digits, k=50000))
    long_pattern = ''.join(random.choices(string.ascii_lowercase + string.digits, k=100))
    cases.append((long_text, long_pattern))
    
    return cases
def main():
    for i, (text, pattern) in enumerate(test_cases(), 1):
        print(f"Test Case {i}: Text(len={len(text)}), Pattern(len={len(pattern)})")

        # Run all algorithms
        results = {}
        results['Your BM Opt'] = boyer_moore_optimized(text, pattern)
        results['Standard BM'] = boyer_moore_standard(text, pattern)
        results['Built-in'] = builtin_find_all(text, pattern)
        results['KMP'] = kmp_search(text, pattern)
        results['Rabin-Karp'] = rabin_karp_search(text, pattern)

        # Check correctness vs Built-in
        for algo, res in results.items():
            print(f"  {algo}: Correct? {res == results['Built-in']}")

        # Benchmark time (less repeats for large text)
        repeat = 1000 if len(text) < 1000 else 100
        for algo, func in [('Your BM Opt', boyer_moore_optimized),
                           ('Standard BM', boyer_moore_standard),
                           ('Built-in', builtin_find_all),
                           ('KMP', kmp_search),
                           ('Rabin-Karp', rabin_karp_search)]:
            t = benchmark_search(func, text, pattern, repeat)
            print(f"  {algo} avg time: {t:.5f} ms")

        print()

if __name__ == "__main__":
    main()