#import "../../theme/iTalk.typ": *
#import "@preview/algo:0.3.3": algo, i, d, comment, code

// TODO fill all "TODO" with your information

#show: nenu-theme.with(
  short-title: "MWCP",
  short-date: "240911",
  short-author: "Virgil" 
)

#let argmax = math.op("arg max", limits: true)
#let argmin = math.op("arg min", limits: true)

#title-slide(
  title: text("A Semi-Exact Algorithm for Quickly Computing a Maximum Weight  Clique in Large Sparse Graphs", size: 0.8em),
  authors: (
    name: "凌典",
    email: "virgiling7@gmail.com"
  ),
  logo: image("../template/fig/nenu-logo-title.png", width: 30%),
  institution: "Northeast Normal University",
  date: "2024-09-11"
)

#slide(
  title: "Defination",
  session: "Problem"
)[
  给定一个点加权图 $G=<V, E, W>$
    
  期望找到一个团 $C$，使得 $sum_(v in C) w(v)$ 最大，其中，团 $C$ 是 $G$ 的一个完全子图。
  
  // #only((2, 3))[
  //   最大加权团问题的上界，我们可以通过图着色得出：

  //   由于一个合法的图着色是对图 $G$ 的顶点集 $V(G)$ 一个划分 ${A_1, A_2, dots, A_k}$，每个划分都是一个独立集#footnote("独立集中的顶点两两不相邻")，于是，任意一个团最多只有一种颜色的一个顶点
  // ]
  // #only(3)[

  //   我们在下面给出如何寻找最大加权团的上界
  // ]
]

#slide(
  title: "Framework",
  session: "Algorithm"
)[

  #only(1)[
      算法通过两个子算法交叉运行，如@algo1[算法] 所示
      #set text(size: 20pt)
      #figure(
        algo(
          title: "FastWClq",
          parameters: ($G$, $"cutoff"$),
          strong-keywords: true,
          row-gutter: 15pt
        )[
          while $"elapsed time" < "cutoff"$#i\
            while $w(C) lt w(C^*)$ do#i\
              $C arrow.l "FindClique"(G)$#d\
            $C^* arrow.l C$\
            $G arrow.l "ReduceGraph"(G, w(C^*))$\
            if $G eq emptyset.rev$ then#i\
              return $C^*$ #comment[最优解] #d\
            return $C^*$ 
        ]
      )<algo1>
  ]

  #only(2)[
    #set text(size: 20pt)
    #grid(
      columns: 1,
      row-gutter: .5em,
      rows: (1.8fr, .6fr),
      algo(
        title: "FastWClq",
        parameters: ($G$, $"cutoff"$),
        strong-keywords: true,
        row-gutter: 10pt
      )[
        while $"elapsed time" < "cutoff"$#i\
          while $w(C) lt w(C^*)$ do#i\
            $C arrow.l "FindClique"(G)$#d\
          $C^* arrow.l C$\
          $G arrow.l "ReduceGraph"(G, w(C^*))$\
          if $G eq emptyset.rev$ then#i\
            return $C^*$ #comment[最优解] #d\
          return $C^*$ 
      ],
      text(size: 22pt)[
          其中，规约图的算法本质上是去除了一些 #pin(5) 不可能在最优解中 #pin(6) 的顶点，因此，当图被规约完（也就是不存在能被规约的顶点了），我们就找到了最优解

          这就是为什么算法被称为半精确

      ]
    )
    #pinit-highlight(5, 6)
  ]

]

#slide(
  title: "FindClique",
  session: "Algorithm"
)[

  #only(1)[
    我们首先规定几个记号：

    - $"StartSet"$：初始的顶点集合，表示从哪个点开始构造团，最开始时为 $V$
    - $"CandSet" = sect.big_(v in C)N(v)$：候选集，由当前团中所有顶点的的公共邻居组成
    
    我们还需要规定 $v$ 的有效邻居 $u$ 为 ${u | u in N(v) sect "CandSet"}$
  ]

  #only(2)[
    整体的算法如@algo2[算法] 所示
    #set text(size: 17pt)
    #figure(
        algo(
          title: "FindClique",
          parameters: ($G$, $"StartSet"$),
          strong-keywords: true,
          row-gutter: 10pt
        )[
          if $"StartSet" eq emptyset.rev$ then#i\
            $"StartSet" arrow.l  V$\
            $t arrow.l t gt "MAX_BMS" ？ "MIN_BMS" : 2 times t$#d\
          $u^* arrow.l "random"("StartSet")$\
          while $"CandSet" eq.not emptyset.rev$ do#i\
            $v arrow.l "ChooseSolutionVertex"("CandSet", t)$\
            if $w(C) + w(v) + w(N(v) sect "CandSet") lt w(C^*)$ then#i\
              break #comment[剪枝] #d\
            $C arrow.l C union {v}$\
            $"CandSet" arrow.l "CandSet" \\ {v} sect N(v)$#d\
          if $w(C) gt.eq w(C^*)$ then#i\
            $C arrow.l "ImproveClique"(C)$#d\
        ]
      )<algo2>
  ]

  #only(3)[
    我们通过动态 `BMS`(`Best from Multiple Selection`) 策略来进行选点，`BMS` 是一种概率启发式策略，通过多次采样，最后选择采样中表现最好的。

    因此，在介绍策略之前，我们首先需要定义打分函数，用于评判顶点的好坏
  ]

  #only((4, 5, 6))[

    #only(4)[
      我们定义贡献值为 $"benfit"(v) = w(C_f) - w(C)$，其中，$C_f$ 是最终由 $C union {v}$ 得到的团。然而，我们无法直接计算 $"benfit"(v)$，因为 $C_f$ 是未知的。
    ]
    
    因此，我们在这里使用上下界的方法来逼近 $"benfit"(v)$

    #only((5, 6))[
      
      1. 当 $v$ 被加入到 $C$ 中后，权重增长的一个显然的下界为 $w(v)$
      2. 当 $v$ 被加入后，最好的可能是 $v$ 的所有有效邻居都被加入到 $C$ 中，也就是说 $C arrow.l C union {v} union (N(v) sect "CandSet")$，这个时候，我们得到了上界为 $w(v) + w(N(v) sect "CandSet")$
      
      #only(6)[
        于是，我们通过上下界的平均#footnote[这里的 $2$ 可以被替换成其他数值，可以认为是一个可调参数]来作为顶点的分数：

        $
          hat(b) = w(v) + w(N(v) sect "CandSet") / 2
        $
      ]
    ]
  ]

  #only((7, 8))[
    #only(7)[
      有了打分函数后，`BMS` 选择顶点的算法如@algo3[算法] 所示：
      #set text(size: 18pt)
      #figure(
        algo(
            title: "ChooseSolutionVertex",
            parameters: ($"CandSet"$, $t$),
            strong-keywords: true,
            row-gutter: 15pt
          )[
            if $|"CandSet"| lt t$ then#i\
              return $argmax_(hat(b)(v)) "CandSet"$#d\
            $v^* arrow.l "random(CandSet)"$\
            for $i arrow.l 1$ to $t - 1$ do #i\
              $v arrow.l "random(CandSet)"$\
              if $hat(b)(v) gt hat(b)(v^*)$ then#i\
                $v^* arrow.l v$#d#d\
            return $v^*$
          ]
        )<algo3>
    ]
    #only(8)[
        #set text(size: 18pt)
        #algo(
          title: "ChooseSolutionVertex",
          parameters: ($"CandSet"$, $t$),
          strong-keywords: true,
          row-gutter: 15pt
        )[
          if $|"CandSet"| lt t$ then#i\
            return $argmax_(hat(b)(v)) "CandSet"$#d\
          $v^* arrow.l "random(CandSet)"$\
          for $i arrow.l 1$ to $t - 1$ do #i\
            $v arrow.l "random(CandSet)"$\
            if $hat(b)(v) gt hat(b)(v^*)$ then#i\
              $v^* arrow.l v$#d#d\
          return $v^*$
        ]
      我们可以通过控制 $t$ 的大小来控制贪心的程度
      
      而称为动态的原因就是因为在运行的过程中，每当 $"StartSet"$ 变空后，$t$ 的大小会动态改变  
    ]
  ]
]

#slide(
  title: "Improving the Clique", 
  session: "Algorithm"
)[
  #only(1)[
    #set text(size: 18pt)
    #algo(
      title: "FindClique",
      parameters: ($G$, $"StartSet"$),
      strong-keywords: true,
      row-gutter: 10pt
      )[
        if $"StartSet" eq emptyset.rev$ then#i\
          $"StartSet" arrow.l  V$\
          $t arrow.l t gt "MAX_BMS" ？ "MIN_BMS" : 2 times t$#d\
        $u^* arrow.l "random"("StartSet")$\
        while $"CandSet" eq.not emptyset.rev$ do#i\
          $v arrow.l "ChooseSolutionVertex"("CandSet", t)$\
          if $w(C) + w(v) + w(N(v) sect "CandSet") lt w(C^*)$ then#i\
            break #comment[剪枝] #d\
          $C arrow.l C union {v}$\
          $"CandSet" arrow.l "CandSet" \\ {v} sect N(v)$#d\
        if $w(C) gt.eq w(C^*)$ then#i\
          #pin(1)$C arrow.l "ImproveClique"(C)$#pin(2)#d\
          #pinit-highlight(1, 2)
      ]
  ]

  #only(2)[
    我们的做法是：$forall v in C$，我们尝试移除 $v$，于是，$"CandSet"$ 变为 $N(v) sect sect.big_(u in C\\{v})N(u)$
    
    我们尝试检查，是否存在一个 $v^prime in "CandSet"$，使得 $C \\ {v} union {v^prime}$ 优于 $C$，是的话更新解，否则保持 $C$
  ]
]


#slide(
  title: "Reduction",
  session: "Algorithm"
)[
  如果我们能通过精确的方法（分支限界）来规避掉一些绝不可能出现在最优解的顶点，那么就能够加快求解的速度，这里我们通过一个简单的上界方法进行限界。
]

#slide(
  title: "Upper Bound",
  session: "Algorithm"
)[
  #only((1, 2, 3))[
    给定一个点加权图 $G=<V, E, W>$，对于一个顶点 $v in V$，假设 $"UB"(v)$ 表示 #pin(1) 任意一个包含顶点 $v$ 的团 $C$ 的上界 #pin(2)
    
    显然，上界一定会满足以下性质

    $"UB"(v) gt.eq max{w(C) | "C is a clique of " G, v in C}$

    #pinit-highlight(1, 2)
  ]

  #only((2, 3))[
    于是，我们可以通过这个规则，来去除一些不满足这个性质的顶点，也就是不在 #pin(3)最优解中的顶点 #pin(4)

    #pinit-highlight(3, 4)
  ]

  #only(3)[
      给定一个团 $C$，我们通过下面三种方法来计算其上界
  ]

  #only((4, 5))[
    对于任意一个包含了顶点 $v$ 的团 $C$，我们都有 $"UB"_0(v) = w(N[v])$#footnote[$N[v]$ 表示顶点$v$ 的领域闭集，也就是 ${u | <u, v> in E} union {v}$] 
  ]

  #only(5)[

    由于 $C$ 是一个完全图，因此如果 $v in C$，那么 $forall u in N(v), u in C$，所以我们可以通过计算 $N[v]$ 的权重来得到 $C$ 的上界#footnote[因为 $|C| lt.eq |N[v]|$]。
  ]

  #only((6, 7))[
    
  ]

]

