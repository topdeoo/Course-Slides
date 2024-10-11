#import "@preview/elsearticle:0.2.1": *

#show: elsearticle.with(
  title: [`NAE-SAT` Survey],
  authors: (
    (
      name: "Virgil",
      affiliation: "NENU",
      corr: none,
      id: "1",
    ),
  ),
  journal: "Notebook '24",
  keywords: ("NAE-SAT", "MAX NAE-SAT"),
  format: "review",
)

#set quote(block: true)
#show quote.where(block: true): it => {
  text(font: "Segoe Script")[“] + it.body + text(font: "Segoe Script")[”] + "\n"
  if it.attribution != none {
    set align(right)
    set text(font: "Segoe Script")
    "-" + it.attribution
  }
}

= NAE-SAT Definition

#quote(attribution: [WikiPedia])[
  In computational complexity theory, the set splitting problem is the following decision problem: given a family F of subsets of a finite set S, decide whether there exists a partition of S into two subsets S1, S2 such that all elements of F are split by this partition, i.e., none of the elements of F is completely in S1 or S2. Set Splitting is one of Garey & Johnson's classical NP-complete problems. The problem is sometimes called hypergraph 2-colorability.
]

NAE-SAT 是

@porschenXSATNAESATLinear 对于 NAE-SAT，到目前为止还没有取得这样的进展，这并不奇怪，因为对于不受限制的情况，NAE-SAT 与 SAT 本身一样难 [22， 23]。因此，我们面临的问题是，是否可以提供精确的确定性算法来分别解决 NAE-SAT，LCNF 上的 SAT 比在 n 个变量上的输入实例上的 2n 步更快。

#bibliography("ref.bib")c:\Users\Virgil\Documents\playground\TypstGround\Survey\ref.bib