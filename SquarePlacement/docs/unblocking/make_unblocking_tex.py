"""Builds unblocking.tex - "Unblocking", the rule as implemented (the clean
split, 2026-11-16) against suspending and against unblocking inside the
pass. The figures are computed: swept map seed 1 (superlattice placement
tiles), one pass, then the CPU's unblocking step (Pyramid.analyse_and_unblock).
Run: python make_unblocking_tex.py, then pdflatex unblocking.tex (twice)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "foundation"))

from cores import DeviceMemory, Pyramid                          # noqa: E402
from map_of_squares import StateEnum                             # noqa: E402
from test_nary_superlattice import completable_swept_map, superlattice_tiles   # noqa: E402
from test_nary_isolation import _state_only                      # noqa: E402
from make_foundation_tex import tikz_map, PREAMBLE as FOUNDATION_PREAMBLE      # noqa: E402

CODE = {StateEnum.free: 0, StateEnum.chosen: 1, StateEnum.blocked: 2}
ROWS, COLS = [5, 13], [8, 18]


def key(p):
    return f"{p[0]},{p[1]}"


def grid(states, shape):
    return [[CODE[states[(r, c)]] for c in range(shape[1])] for r in range(shape[0])]


def fig(st, **kw):
    d = dict(st=st, rows=ROWS, cols=COLS, cells={}, labels={}, arrows=[], rings=[], pixels=[], blocks=[])
    d.update(kw)
    return d


def mark(d, cells, cls):
    for p in cells:
        d["cells"].setdefault(key(p), []).append(cls)


def build():
    raw, _filled = completable_swept_map(7, 3, 1)
    Pyramid.set_placement_tiles(superlattice_tiles(7, 7, (1, 1)), 3, 0)
    mem = DeviceMemory(_state_only(raw), 3)
    Pyramid.onepass(mem)
    report = dict(mem.refuted)
    rounds = Pyramid.analyse_and_unblock(DeviceMemory(_state_only(raw), 3))
    mem2 = DeviceMemory(_state_only(raw), 3)
    Pyramid.analyse_and_unblock(mem2)
    (before, refuted, seats), = mem2.unblock_rounds
    shape = raw.shape
    st0 = grid(before, shape)
    after = {idx: mem2.map[idx].state for idx in before}
    st1 = grid(after, shape)

    F = {}
    # 1: the pass's report for (9,15) and (11,15)
    d = fig(st0)
    c915, p915 = report[(9, 15)]
    c1115, _p = report[(11, 15)]
    mark(d, [x for x in c915 | c1115 if x not in ((9, 15), (11, 15))], "alias")
    mark(d, [(9, 15), (11, 15)], "cand")
    d["labels"]["9,15"] = "Z"
    d["rings"] = [list(p) for p in p915]
    F["report"] = d
    # 2: (9,11), refuted through its n-ary facts
    d = fig(st0)
    c911, p911 = report[(9, 11)]
    mark(d, [x for x in c911 if x != (9, 11)], "alias")
    mark(d, [(9, 11)], "cand")
    d["rings"] = [list(p) for p in p911]
    F["report911"] = d
    # 3: after the CPU's unblocking step
    d = fig(st1)
    mark(d, sorted(refuted), "newblocked")
    mark(d, seats, "focus")
    newly = [idx for idx in before if before[idx] == StateEnum.free and after[idx] == StateEnum.blocked
             and idx not in refuted]
    mark(d, newly, "ghost")
    F["after"] = d
    facts = dict(rounds=rounds, refuted=sorted(refuted), seats=seats, newly=sorted(newly),
                 report={c: (len(ch), sorted(p)) for c, (ch, p) in report.items()})
    return F, facts


PREAMBLE = FOUNDATION_PREAMBLE.split(r"\title{")[0] + r"""
\title{\textsf{Unblocking}\\[0.3em]\large the rule as implemented, against suspending and against unblocking inside the pass}
\date{16 November 2026}
\author{}
"""

BODY = r"""
\begin{document}
\maketitle
\thispagestyle{empty}

A candidate $Z$ is \emph{refuted} when choosing it would force two cells that block each other: two diagonal neighbours, both to be chosen. This note describes what the algorithm now does with such a candidate, why that is sound, and how it compares with the two earlier approaches: suspending $Z$ for a pass (until 15 November), and unblocking inside the pass (15 to 16 November). Section numbers of the form ``F\,9'' refer to \emph{Conditions and the Map} (\code{docs/foundation}).

%------------------------------------------------------------------
\section{The situation}

\begin{figure}[H]\centering
@@FIG:report@@\hspace{0.8cm}@@FIG:report911@@
\caption{Swept map seed 1, superlattice placement tiles, as one pass reports it. Left: $Z=(9,15)$ forces the seats $(8,15)$ and $(7,16)$, which are diagonal neighbours (red); $(11,15)$ (also outlined) forces $(9,15)$'s cells and runs into the same pair. Right: $(9,11)$ is refuted only through its n-ary facts; its chain up to the first contradiction has @@n911@@ cells and two blocking pairs.}
\end{figure}

One pass on this map reports three refuted candidates: $(9,15)$, $(11,15)$ and $(9,11)$. Every chain is taken up to its \emph{first} contradiction only (\code{Base.chain\_until\_contradiction}, and the n-ary step stops there too): whatever would follow from placing two cells that block each other is worthless.

%------------------------------------------------------------------
\section{Why a refutation is permanent}

Forcing is a logical consequence of the current state $S$: ``$Z$ chosen'' together with $S$ implies every cell of $Z$'s chain. More placements only add to $S$: conditions shrink (a chosen member drops out) or die (a blocked member), and nothing derived from $S$ is ever withdrawn. So if $Z$ and $S$ lead to two chosen diagonal neighbours, which is a pixel no cell can cover (F\,4, F\,9), then so do $Z$ and every later state. \textbf{No completion of the map contains $Z$.}

Concretely for $(9,15)$: if $(8,15)$ is later chosen from elsewhere, it blocks $(7,16)$, which $(9,15)$ still forces; if $(7,16)$ gets blocked, $(9,15)$ forces a blocked cell. Either way it is the hole again. The earlier assumption that ``once one of the blocked cells is chosen, $Z$ becomes eligible again'' (12 November) is wrong.

Hence the \textbf{unblocking rule}: a refuted candidate is blocked. In the algebra this is the size-1 defect term $\T{Z}\to\bot$ (F\,9); its effect is the one a chosen diagonal neighbour has: every condition naming $Z$ dies. It stands beside the \textbf{seat rule} (a block with three corners blocked and one free: the free cell is chosen). Both turn a fact that holds in every completion into a state, and both are applied by the CPU between passes.

\paragraph{No freedom is lost.} Blocking $Z$ removes no valid completion, since none contains $Z$. A seat the block creates is chosen in every completion too: with $Z$ out of every completion, the block's last free corner is its only cover. The set of valid completions is the same before and after an unblocking step.

%------------------------------------------------------------------
\section{As implemented: the clean split}

The pass is \emph{analysis only}: it answers ``if $Z$ is chosen, this will happen'' and never changes a cell's \code{.state}. Acting on its findings is the CPU's placement step.

\begin{algebra}\raggedright
\textbf{Pass} (\code{Pyramid.onepass}), on a fixed state:\\
\quad 1.\ \code{init\_map}: conditions from \code{.state}\\
\quad 2.\ refuted from single links $\Rightarrow$ suspended (\code{Base.suspend\_contradictory})\\
\quad 3.\ up and down passes with round trips; the n-ary step stops a candidate at its first contradiction\\
\quad 4.\ refuted from n-ary facts (\code{mem.nary\_contradictory}, \code{Base.contradictory\_after\_nary})\\
\qquad $\Rightarrow$ suspended from the start, pass rerun from 1\\
\quad 5.\ report: \code{mem.refuted} $=\{Z: (\text{chain},\ \text{blocking pairs})\}$\\[0.4em]
\textbf{CPU placement step} (\code{Pyramid.unblock}):\\
\quad 6.\ unblocking rule: every reported $Z$ gets \code{.state = blocked}\\
\quad 7.\ seat rule: fill seats to a fixed point (\code{Pyramid.fill\_seats}); a real block\\
\qquad with four blocked corners raises \code{Uncompletable}\\
\quad 8.\ log the round in \code{mem.unblock\_rounds}; next pass\\[0.4em]
\cmt{\code{Pyramid.analyse\_and\_unblock} alternates the two until a pass refutes nothing.}
\end{algebra}

Within the pass a refuted candidate is merely \emph{suspended}: it is no candidate, starts no chain, and its own \code{.blocked} entry is $(Z,Z)$. Its \code{.state} stays free. Suspension is now only the pass's way of not using what it has just shown impossible; it is lifted by the CPU step, which blocks $Z$ for good.

The next \code{init\_map} needs nothing new: a blocked cell with no chosen cell behind it is already handled (\code{Base.blocked\_corner\_conditions}): conditions naming $Z$ are dropped, and the blocks with $Z$ as a corner get conditions for their remaining open corners.

\begin{figure}[H]\centering
@@FIG:after@@
\caption{After the CPU's unblocking step. Red outline: the three refuted candidates, now blocked. Blue outline: the seats this creates, $(10,12)$ and $(10,14)$, now chosen. Dashed: the cells those seats block. The next pass on this state refutes nothing.}
\end{figure}

\paragraph{Results.}
\begin{center}\small
\begin{tabular}{@{}p{0.27\linewidth}p{0.07\linewidth}p{0.3\linewidth}p{0.28\linewidth}@{}}
\toprule
map & round & blocked & seats filled\\
\midrule
swept seed 1, superlattice & 1 & $(9,11)$, $(9,15)$, $(11,15)$ & $(10,12)$, $(10,14)$\\
swept seed 2, superlattice & 1 & $(10,11)$, $(11,3)$, $(16,15)$, $(16,17)$, $(17,11)$ & $(17,14)$, $(17,16)$, $(18,12)$, $(18,16)$\\
 & 2 & $(15,11)$ & --\\
swept seed 2, \code{build\_swept\_map} layout & 1 & $(16,15)$, $(16,17)$ & $(17,14)$, $(17,16)$, $(18,12)$, $(18,16)$\\
frozen area & -- & -- & --\\
\bottomrule
\end{tabular}
\end{center}
On seed 2 with superlattice tiles, $(15,11)$ is refuted only on the state the first round leaves: a refutation that needs the blocks and seats of an earlier round.

%------------------------------------------------------------------
\section{Approach 1: suspending (12 to 15 November)}

The refuted candidate was suspended for the pass and nothing else happened. The state stayed as it was, so every later pass found the same refutation again and suspended $Z$ again. The idea behind it was a ``harmless postponement'': once one of the blocking cells is chosen by another placement, $Z$ would become eligible again. Section 2 shows that this never happens.

\begin{itemize}[leftmargin=*,itemsep=0.2em]
\item \textbf{Sound, but it keeps information out.} No pass ever learns that $Z$ is impossible as a fact of the map: the conditions naming $Z$ stay alive, the blocks with $Z$ as a corner never get their conditions, and the seats that blocking $Z$ makes, $(10,12)$ and $(10,14)$ here, are never found to be forced.
\item \textbf{Repeated work.} The refutation is derived anew in every pass, including the n-ary ones like $(9,11)$, which cost a rerun of the whole pass each time.
\item \textbf{Same completions.} Nothing invalid is ever chosen, since a suspended $Z$ is never a decision; but the map does not converge to a state where the analysis finds nothing left to refute.
\end{itemize}

%------------------------------------------------------------------
\section{Approach 2: unblocking inside the pass (15 to 16 November)}

The first implementation applied the unblocking rule \emph{inside} \code{onepass}: a refuted candidate was blocked at once, the seats that made were filled, and the pass restarted from \code{init\_map} on the new state, until a pass refuted nothing.

\begin{itemize}[leftmargin=*,itemsep=0.2em]
\item \textbf{The analysis placed cells.} Filling $(10,14)$ is a placement, made in a step meant to answer ``if $Z$ is chosen, this will happen''. Forced or not, the analysis then reported results for a state the caller had not handed it. On the GPU this would mean the device writing \code{.state}, which the design leaves to the CPU.
\item \textbf{Failure in the wrong place.} On the raw swept map with seed 0, whose open seats cannot all be filled, the analysis itself raised \code{Uncompletable}, because it filled those seats. A map problem surfaced as an analysis failure.
\item \textbf{More information within a call, in smaller steps.} Each refutation found before the up pass was blocked before the rest was analysed. On seed 1, $(9,15)$ and $(11,15)$ were blocked first; on that state $(9,11)$ was refuted by its single links alone (pair $(10,13)/(11,12)$), one round later. That needed two rounds where the clean split needs one.
\item \textbf{Checking got harder.} A refutation found in a later round could rest on an earlier round's blocks, so every check had to replay each round on its own state.
\end{itemize}

%------------------------------------------------------------------
\section{Side by side}

\begin{center}\small
\begin{tabular}{@{}p{0.23\linewidth}p{0.22\linewidth}p{0.22\linewidth}p{0.22\linewidth}@{}}
\toprule
 & suspending & unblocking in the pass & clean split (implemented)\\
\midrule
refuted $Z$ & no candidate for the pass & blocked at once & no candidate for the pass; reported\\
\code{.state} changed by the pass & no & yes: blocks and seats & no\\
who blocks $Z$ & nobody & the pass & the CPU's placement step\\
seats after blocking & never filled & filled by the pass & filled by the CPU step\\
refutation found again & every pass & no & no\\
same valid completions & yes & yes & yes\\
learns from the blocks & never & within the same call & from the next pass on\\
seed 1 & 3 suspended each pass & 2 rounds & 1 round\\
\code{.blocked} after the pass & pairs and $(Z,Z)$ & pairs only & pairs and $(Z,Z)$\\
\code{Uncompletable} raised in & nowhere & the analysis & the CPU step\\
\bottomrule
\end{tabular}
\end{center}

%------------------------------------------------------------------
\section{Open points}
\begin{enumerate}[leftmargin=*,itemsep=0.3em]
\item \textbf{Blocking pairs are not unblocked.} Two candidates from different placement tiles that block each other, like $(3,5)$ and $(5,9)$ on seed 1, are each possible alone; only the pair is excluded. They stay in \code{.blocked} and constrain the decisions, not the state.
\item \textbf{Cost of the split.} What a block teaches is used one pass later. Since the CPU step is cheap and the next pass is needed anyway after the decisions, this is expected to cost little; it has not been measured.
\item \textbf{The decisions.} \code{analyse\_and\_unblock} has no decisions yet. In the full CPU main, the unblocking and seat rules go into the same placement step as the chosen candidates, and the order within that step needs fixing: block refuted candidates before or after placing the decisions, which may refute more.
\end{enumerate}

\end{document}
"""


def main():
    F, facts = build()
    print(facts)
    body = BODY
    for name, f in F.items():
        body = body.replace(f"@@FIG:{name}@@", tikz_map(f, S=0.38))
    body = body.replace("@@n911@@", str(facts["report"][(9, 11)][0]))
    assert "@@" not in body
    open(os.path.join(HERE, "unblocking.tex"), "w", encoding="utf-8").write(PREAMBLE + body)
    print("written")


if __name__ == "__main__":
    main()
