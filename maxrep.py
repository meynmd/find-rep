from collections import defaultdict

def maxrep(text):
    sa = sorted(
        [text[i:] for i in range(len(text))],
        cmp=compare_ss,
    )
    # rep = [lcp(sa[i], sa[i + 1]) for i in range(len(sa) - 1)]
    rep = defaultdict(int)
    for i in range(len(sa) - 1):
        p = lcp(sa[i], sa[i + 1])
        if p is not None:
            rep[tuple(p[0])] += 1

    # return sorted(
    #     [r for r in rep if r is not None],
    #     key=lambda r: r[1],
    #     reverse=True
    # )
    s = sorted(
        rep.items(), key=lambda x: x[1] * len(x[0]), reverse=True
    )
    return [list(ss) for ss, occ in s if len(ss) > 1]

def compare_ss(s1, s2):
    p = lcp(s1, s2)
    if p is not None:
        junk, dif_idx = p
    else:
        dif_idx = 0
    sym1, sym2 = s1[dif_idx], s2[dif_idx]
    if (type(sym1) is int or type(sym1) is str) and (type(sym2) is int or type(sym2) is str):
        if sym1 < sym2:
            return -1
        elif sym1 > sym2:
            return 1
        else:
            if len(s1) > len(s2):
                return 1
            elif len(s2) > len(s1):
                return -1
    return 0


def lcp(s1, s2):
    for i in range(min([len(s1), len(s2)])):
        if s1[i] != s2[i]:
            break
    if i > 0:
        return s1[:i], i
    else:
        return None
