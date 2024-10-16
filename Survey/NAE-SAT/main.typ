#import "@preview/elsearticle:0.2.1": *

#show: elsearticle.with(
  title: [`NAE-SAT` Survey],
  authors: (
    (
      name: "Dian Ling",
      affiliation: "NENU",
      corr: "virgiling7@gmail.com",
      id: none,
    ),
  ),
  journal: "Notebook '24",
  keywords: ("NAE-SAT", "MAX NAE-SAT"),
  format: "review",
)

#set quote(block: true)
#show quote.where(block: true): it => {
  set par(leading: 1em)
  set text(style: "italic")
  text(font: "Segoe Script")[“] + it.body + text(font: "Segoe Script")[”] + "\n"
  if it.attribution != none {
    set align(right)
    set text(font: "Segoe Script", style: "italic")
    "-" + it.attribution
  }
}

= NAE-SAT

== Definition

首先，我们重申 SAT 的定义：

/ _Definition 1_: 对于一个给定的 CNF 公式 $c_1 and ... and c_m$，其中 $c_i = or.big^t_j x_t$ 且 $t gt.eq 1, t in ZZ$，是否存在一组赋值 $phi = (x_1, dots, x_n), x_i in {0, 1}$，使得 CNF 成真
\
而 NAE-SAT，全称 _Not-All-Equal SAT_，就是在 SAT 的基础上再加上一条约束：
\
/ _Definition 2_: 给定一个 SAT 问题，要求每个子句 $c_i$ 中至少有一个文字为真且至少有一个文字为假 （也就是说一个子句中不可能所有变量均相等）
\
注意到，NAE-SAT 是对称的，即，我们翻转赋值 $phi$ 中的每个赋值后得到 $phi^prime$，$phi^prime$ 仍然是问题的一个成真赋值。

在文章 @porschenXSATNAESATLinear 中， 其提及了两个典型的问题：

+ Set Splitting, 见 @sec:set-splitting
+ Hypergraph bicolorability，见 @sec:hypr-bicolor

并且，文章 @porschenXSATNAESATLinear 在 *Concluding Remarks and Open Problems* 一节中提出：
对于 NAE-SAT，到目前为止还没有取得这样的进展#footnote[指精确算法的提出与优化]，这并不奇怪，因为对于不受限制的情况，NAE-SAT 与 SAT 本身一样难。
因此，我们面临的问题是，是否可以提供精确的确定性算法来解决 NAE-SAT。

当然，这已经是 2011 年提出的 Open Problem 了，但似乎到今年（2024）为止，还没有论文提出精确算法来求解 NAE-SAT

== Set Splitting<sec:set-splitting>

#quote(attribution: [WikiPedia])[
  In computational complexity theory, the set splitting problem is the following decision problem: given a family F of subsets of a finite set S, decide whether there exists a partition of S into two subsets S1, S2 such that all elements of F are split by this partition, i.e., none of the elements of F is completely in S1 or S2. Set Splitting is one of Garey & Johnson's classical NP-complete problems. The problem is sometimes called hypergraph 2-colorability.
]

此问题可以轻易编码为 NAE-SAT：

给定子集簇 $cal(F) = {S_1, S_2, S_3, dots, S_n }$，全集为 $S$， 满足 $forall S_i in cal(F), S_i subset.eq S$，问是否存在两个集合 $X_1, X_2$ 满足：

$
  forall S_i in cal(F), not (S_i subset.eq X_1) and not (S_i subset.eq X_2)
  \
  X_1 sect X_2 = emptyset.rev, X_1 union X_2 = S
$

我们考虑以下朴素编码：

簇中的每个子集本质上就是一个子句，我们对全集 $S$ 中的每个元素进行编号：$x_1, x_2, dots, x_m$，于是，$"CNF" = and.big_(i = 1)^n (or.big_(x_j in S_i)x_j)$

最后我们得到赋值为 $phi$，赋值为真的变量为集合 $X_1$ 中的元素，否则为 $X_2$ 中的元素

== Hypergraph bicolorability<sec:hypr-bicolor>

/ 超图（Hypergraph）: 一种广义的图，其中边（称为超边）可以连接任意数量的顶点，而不仅仅是两个顶点。

显然 @sec:set-splitting 中对于子集簇 $cal(F)$ 与全集 $S$ 可以直接对应到超图 $cal(H) = <V, E> arrow.r.l.double <S, cal(F)>$

于是，问题可以变为一个 2-color 问题，即只用两种颜色如何将超图完全着色。

== Others

暂时未找到 NAE-SAT 的进展，目前可能原因在于：

+ NAE-SAT 的应用场景少

业内#footnote[业内，但本文引用的大多数论文都是Math与Physics分类的，CS分类也是TCS]大多数的研究都集中在 NAE-SAT 的两个变种上，我们将在
@sec:k-sat, @sec:max-k-sat 中介绍。

在 @el-kadiQuantumApproximateOptimisation2024 中提到，NAE-SAT 在计算复杂性的规约上有着重要的作用，
这或许也能解释，为什么关于 NAE-SAT 的均是理论研究。

// == 应用场景

// 我们想知道，是否有问题能够编码为 `NAE-SAT`（即此问题的应用场景是什么），在 @dingSatisfiabilityThresholdRandom2013 中提及的 @CojaOghlan2011TheCT

= NAE-k-SAT / k-NAE-SAT<sec:k-sat>

/ _Definition 3_: 给定一个 SAT 问题，要求每个子句 $c_i$ 中至少有一个文字为真且至少有一个文字为假，并且每个子句中恰好有 $k$ 个文字

=== Theory

在这个问题上的理论研究显然比一般的 NAE-SAT 要多，例如 @dingSatisfiabilityThresholdRandom2013, @slyLocalGeometryNAESAT2023, @namOnestepReplicaSymmetry2023, @gamarnikPerformanceSequentialLocal2017 等，
然而，在这些文章之中，@dingSatisfiabilityThresholdRandom2013 更集中于探讨 _random d-regular k-NAE-SAT_；
而 @slyLocalGeometryNAESAT2023, @namOnestepReplicaSymmetry2023 更集中于讨论 _random d-regular k-NAE-SAT_ 在物理学中的应用（毕竟作者是 #link("https://scholar.google.com/citations?user=y0U2EaUAAAAJ&hl=en")[Allan Sly]），这是因为 _random k-NAE-SAT_ 在稀疏随机约束求解问题中是最简单的一类了，统计物理学中将这种问题描述为 _replica symmetry breaking_。#footnote[然而，这些文章其实都看不懂，没有物理学的知识导致我不知道这些理论有什么用]

首先，我们介绍什么是 _random d-regular k-NAE-SAT_：
\
/ _Definition 4_: 给定一个随机生成的 CNF，其包含了 $d$ 条子句，且每条子句的长度都为 $k$，是否存在一组赋值 $phi$ 使得每条子句中的文字不全为真
\

@dingSatisfiabilityThresholdRandom2013 的工作明确建立了一个阈值 $d_* equiv d_*(k)$，并证明，当 $d < d_*$ 时，随机生成的 CNF 几乎总是可满足的；而当 $d > d_*$ 时，随机生成的 CNF 几乎总是不可满足的。如果阈值 $d_*$ 恰好为整数，我们表明该问题是可满足的，且其概率远离 0 和 1。

=== Non-Theory

@el-kadiQuantumApproximateOptimisation2024 是为数不多的非理论的工作，QAOA@farhiQuantumApproximateOptimization2014 在最大割问题中早有应用，论文中举例用的就是最大割问题，此问题在 @sec:max-cut 中介绍。

然而，@el-kadiQuantumApproximateOptimisation2024 也并未提及 k-NAE-SAT 的实际应用。
由于 QAOA 本质上是一个近似优化算法，因此没办法保证一定获得最优解，于是，衡量 QAOA 的效果是通过成功率(success ratio)的，如@fig:qaoa-success-ratio 所示：

#figure(
  image("success-ratio-qaoa.png", width: 50%),
)<fig:qaoa-success-ratio>

最后，文章将 QAOA 与传统算法WalkSATlm@caiImprovingWalkSATEffective2015 与魔改 WalkSATlm 得到的 WalkSATm2b2，如@fig:walksatm2b2 所示，进行对比（在这里只对比了中位运行时间）：

#figure(
  image("median-running-time-qaoa.png", width: 50%),
)

使用的实例为随机生成的 $k in {3, dots, 10}$ 的 $2500$ 个 NAE-SAT 实例(保证了一定有解)。

#figure(
  image("walksatm2b2.png", width: 50%),
  caption: "WalkSATm2b2 算法示意",
)<fig:walksatm2b2>

= MAX NAE-SAT<sec:max-k-sat>

/ _Definition 5_: 给定一个 CNF，要求每个子句 $c_i$ 中至少有一个文字为真且至少有一个文字为假，目标是找到一组赋值 $phi$，使得满足条件的子句数量最大化
\

_MAX NAE-SAT_ 及其变种 _MAX k-NAE-SAT_#footnote[此变种问题不再给定义，可以参考@sec:k-sat 中的定义]的工作相较于前两个问题会较多。

== MAX CUT<sec:max-cut>

Max Cut，最大割问题，其定义如下：

/ _Definition 6_: 给定一个无向图 $G = (V, E)$，找到一个顶点集合的划分 $(S, overline(S))$，使得划分在连接 $S$ 与 $overline(S)$ 的边的数量最大化
\
这个问题可以认为是 _MAX k-NAE-SAT_ 的一个特例，即 _MAX 2-NAE-SAT_：其中， $V$ 为全集 $S$，$E$ 为子集簇 $cal(F)$。

对于最大割问题的近似算法，在 @odonnellOptimalSdpAlgorithm2008, @goemansImprovedApproximationAlgorithms1995 中均有介绍。

而如果我们为边设置权重，就能得到 _weighted-MAX-CUT_ 问题。

== MAX Set Splitting

对于 $k gt.eq 3$ 的情况，根据 @sec:set-splitting 中，我们也可以为子集簇$cal(F)$ 中的每个子集设置权重，这样就得到了问题 _MAX Set Splitting_，在 @zhangImprovedApproximationsMax2004, @anderssonBetterApproximationAlgorithms 中均有介绍，但工作均为求出近似比 $alpha$，
与 @sec:max-cut 中提及的文章工作类似。

== MAX k-NAE-SAT

在理论方面，@brakensiekMysteriesMAXNAESAT2024 的工作主要是确定了 $k gt.eq 3$ 中，算法的最优近似比，见 @tbl:ratios

#figure(
  kind: table,
  caption: "Optimal approximation ratios for MAX NAE-{k}-SAT.",
  table(
    align: center,
    stroke: .5pt,
    columns: (1fr, 1fr, 1fr),
    [MAX NAE-{$k$}-SAT], [Optimal ratios], [Algorithm],
    [$k = 2$], [$alpha_(G W) approx 0.8786$], [Goemans-Williamson@goemansImprovedApproximationAlgorithms1995],
    [$k = 3$], [$approx 0.9089$], [@brakensiekMysteriesMAXNAESAT2024],
    [$k gt.eq 4$], [$1 - 1 / (2^(k -1))$], [Random Assignment],
  ),
)<tbl:ratios>

// = Conclusion<sec:conclusion>

#bibliography("ref.bib")