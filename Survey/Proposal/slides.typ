/// This template is written by [@zengls](https://github.com/zengls3186428803)
/// Improved by [@virgil](https://github.com/topdeoo)

#import "@preview/numbly:0.1.0": numbly
#import "@preview/cuti:0.2.1": fakebold
#import "@preview/fletcher:0.5.1" as fletcher: node, edge
#import "@preview/i-figured:0.2.4"
#import "@preview/lovelace:0.3.0": *
#import "@preview/cetz:0.3.1"
#import "@preview/chronos:0.2.0"


#let font-size = (
  初号: 42pt,
  小初: 36pt,
  一号: 26pt,
  小一: 24pt,
  二号: 22pt,
  小二: 18pt,
  三号: 16pt,
  小三: 15pt,
  四号: 14pt,
  中四: 13pt,
  小四: 12pt,
  五号: 10.5pt,
  小五: 9pt,
  六号: 7.5pt,
  小六: 6.5pt,
  七号: 5.5pt,
  小七: 5pt,
)

#let font-family = (
  // 宋体，属于「有衬线字体」，一般可以等同于英文中的 Serif Font
  // 这一行分别是「新罗马体（有衬线英文字体）」、「思源宋体（简体）」、「思源宋体」、「宋体（Windows）」、「宋体（MacOS）」
  宋体: ("Times New Roman", "Source Han Serif SC", "Source Han Serif", "Noto Serif CJK SC", "SimSun", "Songti SC", "STSongti"),
  // 黑体，属于「无衬线字体」，一般可以等同于英文中的 Sans Serif Font
  // 这一行分别是「Arial（无衬线英文字体）」、「思源黑体（简体）」、「思源黑体」、「黑体（Windows）」、「黑体（MacOS）」
  黑体: ("Arial", "Source Han Sans SC", "Source Han Sans", "Noto Sans CJK SC", "SimHei", "Heiti SC", "STHeiti"),
  // 楷体
  楷体: ("Times New Roman", "KaiTi", "Kaiti SC", "STKaiti", "FZKai-Z03S"),
  // 仿宋
  仿宋: ("Times New Roman", "FangSong", "FangSong SC", "STFangSong", "FZFangSong-Z02S"),
  // 等宽字体，用于代码块环境，一般可以等同于英文中的 Monospaced Font
  // 这一行分别是「Courier New（Windows 等宽英文字体）」、「思源等宽黑体（简体）」、「思源等宽黑体」、「黑体（Windows）」、「黑体（MacOS）」
  等宽: ("Courier New", "Menlo", "IBM Plex Mono", "Source Han Sans HW SC", "Source Han Sans HW", "Noto Sans Mono CJK SC", "SimHei", "Heiti SC", "STHeiti"),
)

// 中文缩进
#let indent = h(2em)

// 选项栏
#let checkbox(checked: false) = {
  if checked {
    box(
      stroke: .05em,
      height: .8em,
      width: .8em,
      {
        box(move(dy: .48em, dx: 0.1em, rotate(45deg, reflow: false, line(length: 0.3em, stroke: .1em))))
        box(move(dy: .38em, dx: -0.05em, rotate(-45deg, reflow: false, line(length: 0.48em, stroke: .1em))))
      },
    )
  } else {
    box(
      stroke: .05em,
      height: .8em,
      width: .8em,
    )
  }
}

#let datetime-display-cn-declare(date) = {
  date.display("[year] 年  [month]  月  [day]  日")
}

#let distr(s, w: auto) = {
  block(width: w, stack(dir: ltr, ..s.clusters().map(x => [#x]).intersperse(1fr)))
}

#let cover(
  title: (school: "东北师范大学", type: "研究生学位论文开题报告"),
  author_info: (:),
) = {

  set align(center)
  set text(size: font-size.二号, font: font-family.黑体)
  v(4em)
  fakebold[#title.school #v(.5em) #title.type]

  set text(size: font-size.三号, font: font-family.楷体)
  v(3.5em)

  par(
    justify: true,
    grid(
      columns: (.35fr, 1fr),
      row-gutter: 1.3em,
      column-gutter: 0em,
      align: (center, left),
      distr("论文题目", w: 7em), [：#author_info.title],
      distr("报告人姓名", w: 7em), [：#author_info.name],
      distr("研究方向", w: 7em), [：#author_info.direction],
      distr("学科专业", w: 7em), [：#author_info.major],
      distr("年级", w: 7em), [：#author_info.grade],
      distr("学历层次", w: 7em),
      [：博士生 #checkbox(checked: author_info.level == "博士生")
        #h(1em)硕士生 #checkbox(checked: author_info.level == "硕士生")],

      distr("学位类型", w: 7em),
      [
        ：学术学位 #checkbox(checked: author_info.type == "学术学位")
        #h(1em)专业学位 #checkbox(checked: author_info.type == "专业学位")
      ],

      distr("指导教师", w: 7em), [：#author_info.supervisor],
      distr("培养单位", w: 7em), [：#author_info.unit],
    ),
  )
  set align(left)
  pagebreak()
}

#let command = {
  set page(margin: (top: 2.54cm, bottom: 2.54cm, left: 3.18cm, right: 3.18cm))
  v(1.5em)

  [
    #set align(center)
    #set text(size: font-size.三号, font: font-family.楷体)
    #set par(leading: 1em)
    #fakebold[撰写说明]
  ]

  v(1em)
  set text(size: font-size.四号)
  set par(leading: 1.5em, first-line-indent: 2em, spacing: 1.5em, justify: true)
  [
    1.文献综述应基于选题领域内具有代表性的文献进行，需满足一定的字数要求。博士生：文科不得少于10000字，理科不得少于6000字。硕士生：文科不得少于5000字，理科不得少于3000字。

    2.参考文献是指在开题报告中实际引用的文献。博士生实际引用文献须不少于 50 篇，硕士生实际引用文献须不少于 30 篇。参考文献格式参照学位论文格式要求，建议文中引用文献以脚注形式标注，并在文末按照字母顺序列出所有引用文献。

    3.博士生论文开题时间与学位论文通讯评阅时间间隔原则上不少于 1.5 年，硕士生论文开题时间与学位论文通讯评阅时间间隔原则上不少于 8 个月。
    开题报告审查小组根据开题报告情况，在相应的 #checkbox() 内打号。合格的开题报告，由学院存档并作为毕业审核材料之一。

    4.开题报告审查小组根据开题报告情况，在相应的 #checkbox() 内打 $checkmark.light$ 号。合格的开题报告，由学院存档并作为毕业审核材料之一。

    5.开题报告中的字体字号均用宋体小四，页边距上下20MM,左右25MM，用A4纸打印，于左侧装订成册。

    6.开题结束后，研究生需针对开题中所提问题与建议进行修改，并向学院提交开题报告修订花脸稿。

  ]
  pagebreak()
}

#let empty-par = par[#box()]
#let fake-par = context empty-par + v(-measure(empty-par + empty-par).height)

#let doc(it) = {
  set page(margin: (top: 2cm, bottom: 2cm, left: 2.5cm, right: 2.5cm))
  set text(size: font-size.小四, font: font-family.宋体, lang: "zh")
  set par(leading: 1em, first-line-indent: 2em, justify: true)
  set heading(
    numbering: numbly(
      "{1:一}、",
      "{1:1}.{2}",
      "{1}.{2}.{3}",
    ),
  )
  show heading: it => {

    let title = it.body.text.split("（").first()
    let content = it.body.text.split("（").last()
    if title == "参考文献" {
      content = none
    }
    // TODO 优化这部分显示
    v(0.5em)
    [
      #fake-par
      #set par(leading: 1em, first-line-indent: 0em)
      #if it.level == 1 {
        text(font: font-family.黑体, size: font-size.三号)[
          #fakebold[#counter(heading).display() #title]
        ]
        if content != none {
          text(font: font-family.楷体, size: font-size.四号)[
            （#content
          ]
        }
      } else {
        text(font: font-family.黑体, size: font-size.小三)[
          #counter(heading).display() #title
        ]
      }
    ]
  }


  //! 3. 图片&表格设置
  show heading: i-figured.reset-counters
  show figure: i-figured.show-figure


  show figure.where(kind: table): set figure.caption(position: top)
  set figure.caption(separator: " ")
  show figure.caption: fakebold
  show figure.caption: set text(font: font-family.宋体, size: font-size.五号)

  //! 4. 公式编号
  show math.equation.where(block: true): i-figured.show-equation

  show terms: set par(first-line-indent: 0pt)


  it
}

#let nenu-bibliography(
  bibliography: none,
  full: false,
  style: "gb-7714-2005-numeric",
) = {
  [
    = 参考文献
  ]
  assert(bibliography != none, message: "bibliography 函数不能为空")

  set text(lang: "zh", size: font-size.小四, font: font-family.宋体)

  bibliography(
    title: none,
    full: full,
    style: style,
  )
}

// [!FIXME] 增加 dx, dy 偏移量参数，使得签名能够放在恰当的位置上
#let sign(sign_image: none, date: datetime) = {
  place(right + bottom)[
    指导教师签字：#h(5em) #box(
      sign_image, height: 1.15em
    ) \
    #datetime-display-cn-declare(date)
    #h(3em)
  ]
}

#let review_conclusion(teachers, sign_image: none, date: datetime) = {
  let teacher_table_rows = ()
  for teacher in teachers {
    teacher_table_rows += (teacher.name, teacher.title, teacher.workplace)
  }
  set table(stroke: (x, y) => {
    if y == 0 {
      (
        top: (
          dash: "dashed",
          thickness: 0.5pt,
        ),
        left: (
          thickness: 0.5pt,
        ),
        right: (
          thickness: 0.5pt,
        ),
      )
    } else {
      (top: 0.5pt, bottom: 0.5pt, left: 0.5pt, right: 0.5pt)
    }
  })

  stack(dir: ttb)[
    #table(
      columns: (1.53fr, 1.2fr, 3.56fr),
      rows: (2.2em,) * 10,
      inset: 10pt,
      align: center,
      table.cell(colspan: 3)[审查小组意见],
      table.cell(colspan: 3)[开题报告审查小组成员名单],[姓 名],[职 称],[工 作 单 位],
      ..teacher_table_rows
    )
    #set table(stroke: (x, y) => {
      (top: 0.5pt, bottom: 0.5pt, left: 0.5pt, right: 0.5pt)
    })
    #v(-1.2em)
    #table(
      columns: (4fr)
    )[
      #v(1em)
      #set text(weight: "bold")
      审查结论
      #v(5em)

      #show table: set align(center)
      #table(
        columns: (auto, auto),
        inset: 10pt,
        stroke: none,
        align: left,
        [合格，修改后可以进入学位论文写作阶段], [#box(width: 10pt, height: 10pt, stroke: 0.5pt)],
        [ 不合格，需再次进行学位论文开题报告], [#box(width: 10pt, height: 10pt, stroke: 0.5pt)],
      )

      #v(15em)

      #place(bottom + right)[
        #grid(
          columns: 2,
          rows: 2,
          gutter: 2em,
          [组长签字：], [#box(sign_image)],
          grid.cell(colspan: 2)[
            单位公章：#h(2em)
            #datetime-display-cn-declare(date)
          ],
        )
        #v(1em)
      ]
    ]
  ]
}



//! Start with your configuration here

#cover(
  title: (school: "东北师范大学", type: "研究生学位论文开题报告"),
  author_info: (
    title: "可满足问题中基数约束问题的求解算法研究",
    name: "凌典",
    direction: "人工智能",
    major: "计算机科学与技术",
    grade: "2024 级",
    level: "硕士生",
    type: "学术学位",
    supervisor: "殷明浩",
    unit: "信息科学与技术学院",
  ),
)

#command

#show: doc

//! Start writing here

= 研究背景（分析本选题范畴内尚未得到较好解决的学术或实践难题，阐述选题的缘起与依据）

可满足性问题（SAT）不仅是计算机科学中一项基本且广为人知的研究议题，而且它在理论计算机科学领域中，被认为是首个 NP (Non-deterministic Polynomial，NP) 完全问题 @Cook1971TheCO，其在计算复杂性理论中占据着中心地位 @Arora2009ComputationalCA。SAT 问题在众多领域均显著，尤其是在人工智能 @HandbookSatisfiability2009 与形式验证 @McMillan2003InterpolationAS 等领域，展现出其应用的广泛性和重要性。

然而，尽管 SAT 问题研究已取得了显著进展，一些关键的挑战仍未得到有效解决，特别是在处理包含基数约束的复杂 SAT 问题时。基数约束，即限制变量取值数量的约束，如 “恰有 $k$ 个变量为真”，“至少有 $k$ 个变量为真”，或 “至多有 $k$ 个变量为真”，在多个实际应用场景中频繁出现，例如日程安排 @Ach2012CurriculumbasedCT、调度优化 @McAloon1997SportsLS 以及离散事件系统诊断 @grastienDiagnosisDiscreteEventSystems 等。

当前主要的解决策略包括将基数约束转换成子句的不同编码方法，如 Sequential-counter @Sinz2005TowardsAO, Tree-based @JabbourSS13AP 以及 Sort-based @Ach2009CardinalityNA。这些方法虽然在某些情况下有效，但通常会导致子句和变量数量急剧增加，从而大幅降低求解器的效率，并可能使求解过程在有限时间内难以完成。

#pagebreak()
= 文献综述（系统梳理本选题相关的具有代表性的文献，分析相关研究的发展脉络与进展，评述已有研究存在的问题与不足）

在过去的几十年中，求解布尔可满足性(SAT)问题一直是计算机科学领域的核心研究课题。研究者们提出了丰富多样的求解方法，这些方法主要分为精确求解和局部搜索求解两大类。每类方法都有其独特的优势和适用场景，共同推动了SAT求解技术的快速发展。

== 求解 SAT 问题的精确算法

在精确求解方法中，冲突驱动的子句学习(CDCL)算法@MarquesSilva1999GRASPAS @Moskewicz2001ChaffEA @Gomes1998BoostingCS @HandbookSatisfiability2009 被公认为最具突破性的技术之一，在工业级实例和实际应用问题的求解中发挥着重要作用。CDCL算法以DPLL算法 @Davis1960ACP @Davis2011AMP 为基础框架，通过引入一系列创新机制显著提升了求解效率。其核心创新包括：冲突驱动的学习机制能够从搜索失败中获取经验并避免重复错误；定期重启回溯搜索可以摆脱局部困境；学习子句的动态管理策略则平衡了内存占用与推理能力@Beame2004TowardsUA @Jrvisalo2012InprocessingR @Srensson2009MinimizingLC @Wieringa2013ConcurrentCS @van2012satuzk 。这些技术的综合运用使CDCL算法能够有效处理大规模实例，尤其在结构化问题上表现出色。

另一类重要的精确求解方法采用了前瞻技术@Heule2004MarchE @knuthSatisfiablility2018 来优化求解过程，前瞻求解器完全基于 DPLL 算法。这类求解器在做出决策前，通过提前分析变量间的蕴含关系来识别潜在冲突。其核心思想是构建和维护蕴含图，通过图中的路径分析预测可能的矛盾，从而在搜索早期就避免进入无效分支。虽然这种方法在每步决策时需要额外的计算开销，但对于具有特定结构的问题，通过减少搜索空间实现了更高的整体效率。前瞻技术特别适合处理具有密集约束关系的问题，在这类问题上往往能获得显著的性能提升。

== 求解 SAT 问题的局部搜索算法

在局部搜索求解方法中，随机局部搜索(SLS)算法@MarquesSilva1999GRASPAS 以其快速寻解能力而广受关注。不同于精确求解方法的完备性保证，SLS算法通过在解空间中进行智能随机探索，在较短时间内找到满足解。这种方法特别适合处理大规模问题，尤其是在实时系统、在线规划等对响应速度要求较高的场景中具有明显优势。关键在于，SLS算法通过精心设计的启发式策略在局部优化与随机扰动之间取得平衡，既保证搜索的充分性，又避免陷入局部最优。

WalkSAT @MarquesSilva1999GRASPAS 作为最具影响力的SLS算法之一，在求解随机 3-SAT 实例时仍保持着竞争力@Kroc2010AnES @Balint2012ChoosingPD。其做法为，首先对变量进行随机赋值；随后，在每次迭代过程中，随机选择一个未满足的子：如果存在翻转后能增加满足子句数量的变量，则翻转该变量；否则，以概率 $1 - p$ 选择一个最小代价的变量，或以概率 $p$ 随机选择一个变量进行翻转。
然而，随着问题规模的增加，特别是在处理高阶随机 k-SAT 实例($k gt 3$)时，其性能急剧下降。针对这一局限性，研究者们提出了一系列改进算法：sattime@Li2012SatisfyingVF 在随机7-SAT实例上表现出色，probSAT@Balint2012ChoosingPD 在随机5-SAT实例上具有优势，而CCASat@Cai2013LocalSF 则在两种情况下都保持较好性能。
特别值得一提的是 WalkSATlm @caiImprovingWalkSATEffective2015 的改进，它通过引入基于 lmake 的平局打破机制@Cai2013ComprehensiveS @Cai2014ScoringFB @Prestwich2005RandomWW，在随机5-SAT和7-SAT实例上都实现了性能突破，超越了当时最先进的求解器，与完备求解器相比，WalkSATlm 在处理大型实例时通常提供更快的解决方案，尽管它不能保证找到解。

== 基数约束及其编码

在探索高效的SAT求解方法的过程中，研究者们逐渐注意到许多实际问题中蕴含着特殊的约束结构。
例如在约束满足问题（CSP）中，基数约束用于管理满足特定条件的变量数量，是建模现实世界问题的关键，例如资源分配和任务调度。这些约束在整合到布尔可满足性问题（SAT）求解器时，其与伪布尔（PB）约束的关系尤为重要@HandbookSatisfiability2009。
PB 约束提供了一个灵活的框架，用于表达布尔变量的线性组合，使其成为在SAT公式中编码基数约束的强大工具。这种整合显著扩展了SAT求解器在实际问题中的应用范围，如调度、资源分配和配置。
然而，直接将基数约束整合到SAT求解器中面临挑战。简单编码方法可能导致公式规模和复杂度的显著增加，从而影响求解器的效率。

在处理带有基数约束的SAT问题时，研究者们提出了多种CNF编码方法，这些方法主要可以归类为三类：基于计数器的编码、基于排序网络的编码以及基于树的编码。每种方法都有其独特的优点和面临的挑战，尤其是在处理大规模约束时。
首先，基于计数器的编码方法@Sinz2005TowardsAO @JabbourSS13AP，如Sequential Counter，通过构建二进制加法器来跟踪变量取真值的数量。这些辅助变量用于跟踪变量的计数状态，确保在任何情况下都能满足基数约束。
这种编码方法在单元传播方面表现出色，能够有效地减少搜索空间。然而，随着变量数量的增加，辅助变量和子句的数量急剧增长，导致生成的CNF公式过于庞大，从而影响求解效率。这种问题在处理大规模调度问题时尤为明显，即使是最优化的编码方法也可能产生大量的辅助变量和子句。
其次，基于排序网络的编码方法@Ach2009CardinalityNA，如Cardinality Networks，通过将变量进行排序来实现计数。这种方法在某些场景下展现出更好的结构性，可能在特定情况下提高求解效率。然而，与基于计数器的编码类似，随着约束规模的增大，编码复杂度也随之增加，导致求解时间显著延长。
最后，基于树的编码方法@Bailleux2003EfficientCE 利用层次结构来管理约束，可能在效率和可扩展性之间取得平衡。尽管这类方法可能在某些情况下表现良好，但其在处理大规模约束时的性能仍需进一步验证。

这些编码方法在不同问题域中的表现各异@Wynn2018ACO，例如一个变量数为 $n = 66$，子句数 $m = 315$ 带基数约束的 SAT 问题转化为普通 SAT 实例如@tbl:encoding 所示，实验结果表明@Demirovic2014ModelingHS @Demirovic2019TechniquesIB，没有一种编码方法能够在所有情况下占据明显优势。因此，选择适当的编码方法可能需要根据具体问题的特性进行权衡。

#figure(
  table(
    columns: 4,
    table.header(
      [编码方法],
      [变量],
      [子句数],
      [文字数],
    ),

    [Sequential Counter], [1080], [2154], [5358],
    [Tree-based], [328], [1402], [3854],
    [Sort-based], [846], [1296], [3047],
  ),
  caption: [变量与子句扩张],
)<encoding>

== SAT 基数约束问题的求解算法

在面对基数约束编码为 CNF 过程中带来的子句膨胀，我们开始思考，是否能够直接求解基数约束，而不需要转化为 PB 或 CNF 的形式。于是，随着 DPLL(T) 算法的发表@Nieuwenhuis2006SolvingSA，基于 SAT & LCG/SMT 的混合求解方法@abioConflictDirectedLazy2012 引入了冲突驱动的惰性分解策略，巧妙地平衡了约束分解与直接传播两种方法的优势。该策略在处理冲突时，能够根据中间文字的有效性自动切换策略：当中间文字对解决冲突无益时，采用传播方法；而当其有助于解决冲突时，则启用分解方法。这种自适应机制确保了求解效率不会显著低于两种基本方法中的更优者，甚至在某些情况下超越了它们。然而，由于该方法依赖于 DPLL(T) 框架，其在处理大规模基数约束时仍面临挑战，尤其是在扩展性和效率方面表现不足。

而正如上文所述，随机局部搜索(SLS)算法通过在解空间中进行智能随机探索，在较短时间内找到满足解，因此特别适合处理大规模问题，于是，基于 SLS 的 LS-ECNF 方法@Lei2020ExtendedCN 提供了一种全新的视角。
该方法摒弃了传统的编码转换思路，通过扩展合取范式（ECNF）直接表示基数约束，极大地简化了问题表示。ECNF模型通过引入约束数组，自然地描述了基数约束，无需额外的变量或子句，从而降低了问题的复杂性。
此外，LS-ECNF 还设计了适应性评分函数和基于广义单元传播的初始化策略，并在实验中 LS-ECNF 通过与各类编码方式，PB 求解以及基于 DPLL 的精确求解器进行对比，LS-ECNF 在多个维度上超越了现有的 SAT 求解器，证明了专有求解器的必要性。
然而，尽管这些改进显著提升了求解效率，但在处理大型实例时，其约束传播机制仍存在优化空间，可能需要进一步的研究来提高其在大规模问题上的表现。


#pagebreak()
= 研究问题（提出本论文拟回答的核心问题及具体研究问题）

下面，我们形式化的定义基数约束的 SAT 问题。

/ SAT: 在变量集 $X = {x_1, x_2, dots, x_n}$ 上给定一个 CNF 公式， $
  cal(F) = and.big_j^d (or.big_i^k_j l_i), l_i in {x_i, not x_i}, x_i in X
$是否存在变量的一组赋值 $phi = (x_1^prime, x_2^prime, dots, x_n^prime)$ 使得 $cal(F) = 1$ 。

/ 带基数约束的 SAT: 在 SAT 的约束上，新增一条：对于变量集合 $X$，我们有 $
 x_1 + x_2 + dots + x_n \# r, x_i in X
$ 其中，$\# in {lt.eq, gt.eq, eq}$

这里，新增的约束即为：对于变量集合而言，至多/至少/只有 $r$ 个变量取值为真

- “至多 $r$ 个为真” 等价于 “至少 $n - r$ 个为假”
- “只有 $r$ 个为真” 等价于 “至多 $r$ 个为真且至少 $r$ 个为真”

因此我们可以只考虑 $gt.eq_r$ 类型的约束。

更进一步的，我们可以将此约束写为 PB 的形式，如 @eqt:pb-format 所示：

$
  sum_i^k_1 l^((1))_i gt.eq 1 \
  dots.v \
  sum_i^k_d l^((d))_i gt.eq 1 \
  sum_i^n x_i gt.eq r
$<pb-format>

#pagebreak()
= 研究意义（阐述本研究可能的理论贡献与实践价值）

本研究在理论上和实践上均具有重要价值，尤其是在满足性问题（SAT）求解器的理论发展和实际应用领域。

在理论方面，通过结合随机局部搜索（SLS）和完备算法，本研究旨在提高SAT求解器的效率和能力，特别是在处理基数约束方面。这种混合方法可能在解决时间和处理大规模实例方面提供更好的性能指标。此外，研究中可能提出的新型编码方法，有望显著提升求解器在处理复杂约束时的表现，从而为SAT求解器的理论发展做出重要贡献。

SLS 以其在某些问题类型中的高效性而闻名，而完备算法则确保解的最优性。二者的结合可能在速度和准确性之间取得平衡，这对于实际应用中的SAT求解至关重要。

在实践方面，该问题的一大重要应用场景为离散事件系统诊断问题：给定系统模型，如 @fig:des 所示，以及一系列可观测事件序列，如何判断系统是否发生了不可观测的故障事件。

该问题可以建模为一个判定问题：是否存在一个满足特定约束的系统行为序列。而这类问题通常会涉及到大量的基数约束，例如限制某些事件发生的次数或顺序。

#figure(
  fletcher.diagram(
    node-stroke: .1em,
    node-fill: gradient.radial(blue.lighten(80%), blue, center: (30%, 20%), radius: 80%),
    spacing: 4em,
    node((0, 0), `W`, radius: 2em),
    edge((0, 0), (0, 1), `IReboot`, "-|>", bend: 20deg),
    edge((0, 1), (0, 0), `reboot?`, "-|>", bend: 20deg),
    node((0, 1), `WW`, radius: 2em),
    edge((0, 1), (1, 1), `reboot?`, "-|>"),
    node((1, 0), `O`, radius: 2em),
    edge((1, 0), (0, 0), `reboot?`, "-|>"),
    edge((1, 0), (2, 0), `reboot!`, "-|>"),
    node((1, 1), `R`, radius: 2em),
    edge((1, 1), (1, 0), `IAmBack`, "-|>", label-pos: 0.25),
    edge((1, 1), (1, 1), `reboot?`, "--|>", bend: -130deg),
    node((2, 0), `F`, radius: 2em),
    edge((2, 0), (2, 1), `IReboot`, "-|>", label-side: center),
    node((2, 1), `FF`, radius: 2em),
    edge((2, 1), (1, 0), `IAmBack`, "-|>", label-pos: 0.75, label-side: center),
    edge((2, 1), (2, 1), `reboot?`, "--|>", bend: -130deg),
  ),
  caption: "离散事件系统示例",
)<des>

这在制造业、软件诊断、电力网络与交通系统等关键领域具有重要意义。通过改进SAT 求解器的性能，可以更快地检测和隔离故障，减少停机时间，从而在依赖系统在线时间的行业中实现显著的成本节约。

另一个常见的实际问题为离散断层成像问题 @knuthSatisfiablility2018，在这个问题中，我们通过已知的投影来重建离散图像。例如，八皇后问题（如 @fig:eight-queen 所示）就是一个典型的离散断层成像问题，它要求在一个 $8 times 8$ 的棋盘上放置八个皇后，使得没有任何两个皇后互相攻击。这个问题可以通过 @eqt:eight-queen 方程组来表示：

$
  c_j = sum^m_(i = 1)x_(i, j) = 1, 1 lt.eq j lt.eq n\
  r_i = sum^n_(j = 1)x_(i, j) = 1, 1 lt.eq i lt.eq m \
  a_d = sum_(i + j = d + 1)x_(i, j) lt.eq 1, 0 lt.eq d lt.eq m + n \
  b_d = sum_(i - j = d - n)x_(i, j) lt.eq 1, 0 lt.eq d lt.eq m + n\
$<eight-queen>
其中，$n, m$ 分别表示棋盘的列数和行数，$x_(i, j)$ 是一个二元变量，表示在位置 $(i, j)$ 是否放置皇后。
这些方程确保了每行、每列以及每条对角线上最多只有一个皇后，这就是典型的基数约束。离散断层成像问题涉及从已知的投影数据中重建离散图像。
在八皇后问题中，这些投影数据可以理解为每行、每列和每条对角线上皇后的数量。通过这些约束，我们可以唯一地确定皇后的放置位置，这正是离散断层成像的核心思想。

#figure(
  cetz.canvas({
    import cetz.draw: *
    grid((-2, -2), (2, 2), step: 0.5)
    rect((-1.5, 1.5), (-2, 2), fill: black)
    rect((-1, -1.5), (-1.5, -1), fill: black)
    rect((-1, 0), (-0.5, -0.5), fill: black)
    rect((-.5, -2), (0, -1.5), fill: black)
    rect((0, 1), (0.5, 1.5), fill: black)
    rect((0.5, 0), (1, 0.5), fill: black)
    rect((1, -0.5), (1.5, -1), fill: black)
    rect((1.5, 0.5), (2, 1), fill: black)
  }),
  caption: [八皇后问题示例],
)<eight-queen>
\

离散断层成像在实际生活中有着广泛的应用，并且解决这些问题能够带来显著的效果。在医疗成像领域，它用于CT和MRI等技术中，通过提高图像的分辨率和清晰度，帮助医生进行早期疾病诊断，从而制定更有效的治疗方案。在材料科学中，离散断层成像用于分析材料的内部结构，如颗粒分布，这有助于开发高性能材料。在非破坏性测试中，它用于检查产品的内部结构，确保质量的同时不损坏文物或产品。在安全检查方面，如机场行李检查，它提高了检测的准确性和效率，保障了公共安全。在农业中，离散断层成像用于评估植物的健康状况和产量潜力，帮助农民提高作物产量。在地质学中，它用于分析岩石的内部结构，支持资源开发和地质灾害预测。在电子制造业中，它用于检查电路板的内部结构，确保产品质量并减少缺陷率。

#pagebreak()
= 研究设计（针对研究问题，详细阐述本选题的研究内容、基本思路或总体框架、理论基础、具体研究方案等）

本研究设计了一种启发式算法求解带基数约束的 SAT 问题。算法包括以下几个关键步骤：初始化、局部搜索、约束传播和终止条件。
为了方便叙述，我们假设基数约束的集合为 $"card"$。

我们的算法框架如 @fig:cardsat 所示，具体的实现过程有如下具体的规则和方法：

+ *预处理阶段：* 在预处理阶段，我们可以快速得出某些一定成真或一定成假的变量，将其作为单元子句加入到 CNF 中，并维护基数约束集合 $"card"$，从而使得问题规模减小。
+ *初始化阶段：*
+  


1. 初始化阶段，我们随机生成一个满足基数约束的初始解。局部搜索阶段，我们采用启发式搜索策略，通过翻转变量来逐步优化解的质量。约束处理阶段，我们确保在每次变量翻转后，解仍然满足基数约束。终止条件设定为达到预定的迭代次数或找到满足所有约束的解。

#figure(
  kind: "algorithm",
  supplement: [算法],
  pseudocode-list(booktabs: true, numbered-title: [Framework])[
    + *while* elapsed time < cutoff *do*
      + $sigma arrow.l $ a partial assignment with previous solutions
      + $sigma arrow.l $ complete $sigma$ with unit propagation 
      + *while* not_improved $lt $ N *do*
        + *if* $sigma "SAT" cal(F)$ *then*
          + *return* $sigma$
        + $C arrow.l $ an unsat clause chosen with $max$ score
        + *if* $exists x in C "with" max "score"(x)$ *then*
          + $v arrow.l x$
        + *else*
          + $v arrow.l cases(
              "random variable in " C ",   " p,
              "with min cost variable in" C ",  " 1-p
            )$
        + Flip $v$ in $sigma$
        + not_improved $++$
      + *end*
    + *end*
  ],
  caption: "WalkSATlm 算法",
)<cardsat>

在 中，启发式策略用于指导变量的翻转，
具体而言，算法通过评估每个变量翻转对当前赋值的影响，选择最有可能改善当前解的变量进行翻转。
在算法设计时，通常采用以下策略：

+ 最小冲突或最大增益：选择翻转后能减少最多冲突子句数量的变量(选择翻转后能增加最多满足子句数量的变量)，通过减少冲突(增加满足)的子句数，从而更快的获得解。

+ 加权启发式：对于基数约束而言，我们可以为子句分配合适的权重，从而评估变量的翻转效果。

+ 动态调整：根据搜索的进展动态调整启发式策略。例如，在搜索初期可能更倾向于减少冲突子句，而在接近满足的赋值时，可能更倾向于增加满足子句。


在局部搜索中，如何快速检测并跳出局部最优解同样是算法设计的重点。
我们可以采用多启动策略，即在搜索过程中定期重新初始化搜索状态，以期望在新的初始状态下找到更好的解。
最常用的方法是使用 Luby 序列重启，然而 Luby 序列并不是在所有的 SAT 实例上表现都较好。因此，我们也可以通过从历史解决方案中提取的信息来评估部分赋值的质量，获取良好的部分赋值并作为新的部分解@Li2023FGA，其框架如 @fig:cardsat 所示。


#pagebreak()
= 进度安排（按照时间顺序，就研究的进度做出具体的规划）


2024.11 - 2025.01 设计算法及策略

2025.02 - 2025.07 编写代码、调整参数，分析数据

2025.08 - 2025.10 完善算法，进行对比实验测试

2025.11 - 2026.03 撰写、修改毕业论文



#pagebreak()
#nenu-bibliography(bibliography: bibliography.with("main.bib"))


//! 在这里填入自己的签名文件路径
#sign(
  // image("sign.svg", height: 2em),
  date: datetime.today(),
)

// #pagebreak()

// #review_conclusion(
//   (
//     (
//       name: "张三",
//       title: "讲师",
//       workplace: "东北师范大学",
//     ),
//     (
//       name: "李四",
//       title: "教授",
//       workplace: "北京大学",
//     ),
//     (
//       name: "王五",
//       title: "教授",
//       workplace: "复旦大学",
//     ),
//     (
//       name: "赵六",
//       title: "副教授",
//       workplace: "南京大学",
//     ),
//     (
//       name: "孙七",
//       title: "讲师",
//       workplace: "浙江大学",
//     ),
//   ),
//   // image("sign.svg", height: 2em),
//   date: datetime.today(),
// )