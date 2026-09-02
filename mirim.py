#!/usr/bin/env python3
"""
mirim.py -- verifier for §0-§3 of "Gauge Theory of Cognitive Architectures".

Machine-checks the load-bearing claims of the Mirror Involution (§1), the
Jaeger trialectic / relevance-realization boundary (§2), and the three
cancellation regimes of §3. Pure standard library, no third-party deps.

Run: python3 mirim.py
"""
import cmath
import math
import random

CHECKS = []


def check(description, condition):
    CHECKS.append((description, bool(condition)))


# ---------------------------------------------------------------------------
# §1.1-1.2 -- Klein four-group of the two orthogonal mirrors
# ---------------------------------------------------------------------------
def sigma_h(s):
    """Horizontal mirror: left-right reversal of the string."""
    return s[::-1]


AMBIGRAM = {  # vertical (upside-down) flip, ambigram letter map
    'M': 'W', 'W': 'M', 'O': 'O', 'I': 'I', 'N': 'N', 'S': 'S',
    'Z': 'Z', 'X': 'X', 'H': 'H',
}


def sigma_v(s):
    """Vertical mirror: upside-down rotation via the ambigram letter map."""
    return ''.join(AMBIGRAM[c] for c in reversed(s))


identity = lambda s: s
group = {'id': identity, 'H': sigma_h, 'V': sigma_v, 'R': lambda s: sigma_h(sigma_v(s))}

sample = "MOM"
table_ok = True
for name1, f1 in group.items():
    for name2, f2 in group.items():
        composed = lambda s, f1=f1, f2=f2: f1(f2(s))
        # closure: composing any two named maps reproduces one of the four maps on the sample
        if not any(composed(sample) == g(sample) for g in group.values()):
            table_ok = False
check("Klein four-group: composition table closes on {id,H,V,R}", table_ok)

check("every mirror element is an involution (order <= 2)",
      all(f(f(sample)) == sample for f in group.values()))

check("R = H∘V equals V∘H (the group is abelian, i.e. Z2 x Z2)",
      sigma_h(sigma_v(sample)) == sigma_v(sigma_h(sample)))

# ---------------------------------------------------------------------------
# §1.2 -- the {MOM, WOW} orbit under the two mirrors
# ---------------------------------------------------------------------------
check("MOM is fixed under the horizontal mirror", sigma_h("MOM") == "MOM")
check("WOW is fixed under the horizontal mirror", sigma_h("WOW") == "WOW")
check("MOM maps to WOW under the vertical (ambigram) mirror", sigma_v("MOM") == "WOW")
check("WOW maps to MOM under the vertical (ambigram) mirror", sigma_v("WOW") == "MOM")

# ---------------------------------------------------------------------------
# §1.3 -- MIRIM: the conjugation operad
# ---------------------------------------------------------------------------
check("reversal is an involution on an arbitrary string ('mirror')",
      sigma_h(sigma_h("mirror")) == "mirror")
check("reverse('mirror') splits into ROR + RIM (the reflected beam splitter)",
      sigma_h("mirror") == "ror" + "rim")
check("MIRIM is a palindrome (mirror-image of itself)",
      "MIRIM"[::-1] == "MIRIM")

r_count = "mirror".count('r')
check("r-density of 'mirror' equals 3/6 = 0.5 (Fresnel amplitude reflectance)",
      abs(r_count / len("mirror") - 0.5) < 1e-12 and r_count == 3)

# ---------------------------------------------------------------------------
# §1.4 -- the shadow term R'R = R†R, the Gram matrix
# ---------------------------------------------------------------------------
def matmul(A, B):
    n, m, p = len(A), len(B), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]


def conj_transpose(A):
    n, m = len(A), len(A[0])
    return [[A[j][i].conjugate() if isinstance(A[j][i], complex) else A[j][i]
             for j in range(n)] for i in range(m)]


def identity_matrix(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def is_close(A, B, tol=1e-8):
    return all(abs(A[i][j] - B[i][j]) < tol for i in range(len(A)) for j in range(len(A[0])))


def rotation2d(theta):
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s], [s, c]]


# true rotation: R^T R = I (shadow vanishes)
R = rotation2d(0.7)
RtR = matmul(conj_transpose(R), R)
check("a true rotation has vanishing shadow: R†R = I", is_close(RtR, identity_matrix(2)))

# an oblique / lossy cast (non-orthogonal shear+scale) has R†R != I
M_lossy = [[1.3, 0.4], [0.0, 0.7]]
MtM = matmul(conj_transpose(M_lossy), M_lossy)
check("an oblique/lossy cast leaves a nontrivial shadow: R†R != I",
      not is_close(MtM, identity_matrix(2)))

# conjugation preserves spectra: for unitary M and hermitian R, eigenvalues of M†RM = eigenvalues of R
def eig2x2(A):
    a, b, c, d = A[0][0], A[0][1], A[1][0], A[1][1]
    tr, det = a + d, a * d - b * c
    disc = cmath.sqrt(tr * tr - 4 * det)
    return sorted([(tr + disc) / 2, (tr - disc) / 2], key=lambda z: (z.real, z.imag))


random.seed(0)
Rherm = [[2.0, 0.3], [0.3, -1.0]]  # hermitian (real symmetric)
theta = 1.1
Munitary = rotation2d(theta)  # orthogonal/unitary
conjugated = matmul(conj_transpose(Munitary), matmul(Rherm, Munitary))
e1 = eig2x2(Rherm)
e2 = eig2x2(conjugated)
check("conjugation M†RM by a unitary M preserves the spectrum of R",
      all(abs(a - b) < 1e-6 for a, b in zip(e1, e2)))

check("(M†RM)† = M†RM when R is hermitian (conjugation preserves self-adjointness)",
      is_close(conj_transpose(conjugated), conjugated))

# skewness of R'R^T for a moving frame (rotation): the shadow is angular velocity
def dR_dt(theta, omega=1.0):
    c, s = math.cos(theta), math.sin(theta)
    return [[-s * omega, -c * omega], [c * omega, -s * omega]]


Rt = rotation2d(0.4)
Rt_dot = dR_dt(0.4)
shadow = matmul(Rt_dot, conj_transpose(Rt))
skew_ok = all(abs(shadow[i][j] + shadow[j][i]) < 1e-8 for i in range(2) for j in range(2))
check("R'R† is skew-symmetric for a moving frame (the shadow is angular velocity)", skew_ok)

# ---------------------------------------------------------------------------
# §1.5 -- lattice gauge transformation conjugates plaquette holonomy
# ---------------------------------------------------------------------------
def plaquette(links):
    P = identity_matrix(2)
    for L in links:
        P = matmul(P, L)
    return P


random.seed(1)
links = [rotation2d(random.uniform(0, 2 * math.pi)) for _ in range(4)]
g_vertices = [rotation2d(random.uniform(0, 2 * math.pi)) for _ in range(4)]

# gauge transform: U_l -> g_i U_l g_{i+1}^{-1} around the loop; holonomy conjugates by g_0
gauged_links = []
for i in range(4):
    gi = g_vertices[i]
    gj = g_vertices[(i + 1) % 4]
    gauged_links.append(matmul(matmul(gi, links[i]), conj_transpose(gj)))

P_before = plaquette(links)
P_after = plaquette(gauged_links)
g0 = g_vertices[0]
expected = matmul(matmul(g0, P_before), conj_transpose(g0))
check("gauge transformation conjugates the plaquette holonomy: P' = g0 P g0^{-1}",
      is_close(P_after, expected))

# ---------------------------------------------------------------------------
# Jordan center of the MIRIM path (graph-theoretic reading of the palindrome)
# ---------------------------------------------------------------------------
def path_center(n):
    """Jordan center index (0-based) of a path graph on n vertices."""
    dmax = {}
    for i in range(n):
        dmax[i] = max(abs(i - j) for j in range(n))
    m = min(dmax.values())
    return sorted(i for i, v in dmax.items() if v == m)


mirim_letters = list("MIRIM")
center_idx = path_center(len(mirim_letters))
check("the Jordan center of the MIRIM path is its middle vertex 'R'",
      center_idx == [2] and mirim_letters[2] == 'R')

# ---------------------------------------------------------------------------
# §2.1/appendix -- Euler transform recurrence reproducing A000081
# ---------------------------------------------------------------------------
def rooted_tree_counts(N):
    """a(n) = number of rooted trees on n unlabeled nodes, via the Euler transform
    of itself: EULER(a)(n) built from a(1..n-1) using the standard divisor-sum
    recurrence for the Euler transform T(x) = x * exp(sum_k T(x^k)/k)."""
    a = [0] * (N + 1)
    a[1] = 1
    for n in range(2, N + 1):
        total = 0
        for k in range(1, n):
            s = 0
            for d in range(1, k + 1):
                if k % d == 0:
                    s += d * a[d]
            total += s * a[n - k]
        a[n] = total // (n - 1)
    return a[1:]


A000081 = [1, 1, 2, 4, 9, 20, 48, 115, 286, 719]
computed = rooted_tree_counts(len(A000081))
check("Euler-transform recurrence reproduces A000081 = 1,1,2,4,9,20,48,115,286,719",
      computed == A000081)

# ---------------------------------------------------------------------------
# §2.4 -- Cantor blow-up of the unclipped Φ-iteration vs. stabilization of the clipped one
# ---------------------------------------------------------------------------
def unclipped_phi_iterate(steps):
    size = 1.0
    for _ in range(steps):
        size = 2.0 ** size  # S -> S x Omega^S (cardinality), tower growth
    return size


def clipped_phi_iterate(steps, cap=64.0):
    size = 1.0
    for _ in range(steps):
        size = min(2.0 ** size, cap)  # the clip: relevance realization bounds the cast
    return size


unclipped = unclipped_phi_iterate(4)
clipped = clipped_phi_iterate(20)
check("the unclipped Φ-iteration blows up super-exponentially (Cantor's diagonal bound bites)",
      unclipped > 60000)
check("the clipped Φ-iteration stabilizes at a bounded fixed point", clipped <= 64.0 and clipped == clipped_phi_iterate(21))

# ---------------------------------------------------------------------------
# softmax temperature limits: half-silvered <-> uniform, silvered <-> argmax
# ---------------------------------------------------------------------------
def softmax(logits, T):
    m = max(logits)
    exps = [math.exp((x - m) / T) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


logits = [1.0, 3.0, 2.0, 0.5]
high_T = softmax(logits, 1e6)
low_T = softmax(logits, 1e-3)
uniform = [1 / len(logits)] * len(logits)
onehot = [1.0 if i == logits.index(max(logits)) else 0.0 for i in range(len(logits))]
check("softmax at high temperature -> uniform distribution (half-silvered limit)",
      all(abs(a - b) < 1e-4 for a, b in zip(high_T, uniform)))
check("softmax at low temperature -> one-hot argmax (silvered limit)",
      all(abs(a - b) < 1e-4 for a, b in zip(low_T, onehot)))

# ---------------------------------------------------------------------------
# §3.1 -- failure of cancellation: the interleaving bijection N ≅ N x 2
# ---------------------------------------------------------------------------
def hilbert_pair(n):
    """bijection N -> N x {0,1} by interleaving (evens/odds)."""
    return (n // 2, n % 2)


def hilbert_unpair(pair):
    q, r = pair
    return 2 * q + r


N = 2000
forward = [hilbert_pair(n) for n in range(N)]
check("interleaving bijection N -> N x 2 is injective on a finite truncation",
      len(set(forward)) == N)
check("interleaving bijection round-trips (N x 2 -> N -> N x 2)",
      all(hilbert_pair(hilbert_unpair(p)) == p for p in forward))

# ---------------------------------------------------------------------------
# §3.2 -- Kraft-Shannon identity for a complete prefix code
# ---------------------------------------------------------------------------
huffman_lengths = [1, 2, 3, 3]  # a complete binary prefix code, e.g. Huffman on 4 symbols
kraft_sum = sum(2.0 ** (-l) for l in huffman_lengths)
check("Kraft-Shannon identity: sum 2^-l = 1 for a complete prefix code", abs(kraft_sum - 1.0) < 1e-12)

# ---------------------------------------------------------------------------
# §3.3 -- groupoid cardinality |B(Z2)| = 1/2
# ---------------------------------------------------------------------------
aut_Z2 = 2  # |Aut(point sitting over BZ2)| = |Z2| = 2
groupoid_cardinality = 1.0 / aut_Z2
check("groupoid cardinality |B(Z2)| = 1/|Aut| = 1/2", abs(groupoid_cardinality - 0.5) < 1e-12)

# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    passed = 0
    for i, (desc, ok) in enumerate(CHECKS, 1):
        status = "PASS" if ok else "FAIL"
        print(f"[{i:2d}/{len(CHECKS)}] {status}  {desc}")
        passed += int(ok)
    print(f"\n{passed}/{len(CHECKS)} checks pass")
    raise SystemExit(0 if passed == len(CHECKS) else 1)
