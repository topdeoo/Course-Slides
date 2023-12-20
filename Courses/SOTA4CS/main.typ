#import "../theme/iTalk.typ": *

#show: nenu-theme.with(
  short-title: "PyTorch",
  short-date: "23-12-19",
  short-author: "Virgil"
)

#let pinit-highlight-equation-from(height: 2em, pos: bottom, fill: red.darken(5%), highlight-pins, point-pin, body) = {
  pinit-point-from(
    fill: fill, 
    pin-dx: -0.6em, 
    pin-dy: if pos == bottom { 0.8em } else { -0.6em }, 
    body-dx: 0pt, 
    body-dy: if pos == bottom { -1.7em } else { -1.6em }, 
    offset-dx: -0.6em, 
    offset-dy: if pos == bottom { 0.8em + height } else { -0.6em - height },
    point-pin,
    rect(
      inset: .5em,
      stroke: (bottom: 0.12em + fill),
      {
        set text(fill: fill)
        body
      }

    )
  )
}


#title-slide(
  title: "The Hitchhiker's Guide to PyTorch",
  authors: (name: "Virgil", email: "virgiling7@gmail.com"),
  logo: image("fig/nenu-logo-title.png", width: 30%),
  institution: "Northeast Normal University",
  date: "2023-12-19"
)

#slide(
  session: "PyTorch",
  title: "What is PyTorch"
)[
  #only((beginning: 1, until: 5))[
    == Pytorch is a AI Framework
    #parbreak()
    #v(.2em)
    + PyTorch define a grammer to describe the network
    + Use the grammer to build the network
  ]

  #only((beginning: 2, until: 5))[
    == PyTorch is a #pin(1)Compiler#pin(2)
      #pinit-highlight(1, 2)
      #only(3)[
        #pinit-point-from(2)[
          How to understand the compiler \
          here?
        ]
      ]
      #parbreak()
      #only((beginning: 4, until: 5))[
        #pinit-point-from(
          fill: red.darken(5%), 
          pin-dx: -5pt, 
          pin-dy: 5pt, 
          body-dx: -13em, 
          body-dy: 8pt, 
          offset-dx: -45pt, 
          offset-dy: 40pt,
          2
        )[
          #text(fill: red)[How to understand the \
          compiler here?]
        ]
        #only(5)[
          #pinit-point-from(2)[
          As there are many programming \
          languages and hardware, we may \
          develop a all in one compiler
          ]
        ]
    ]
  ]

  #only(6)[
    #raw-render(
      ```dot
        digraph {
          languages -> Frontend
          Frontend -> OPT [label = "LLVM IR"]
          OPT -> Backend [label = "LLVM IR"]
          Backend -> Platform
        }
      ```,
    labels: (
      "languages" : "C/C++/Python/etc",
      "Frontend": "Various Front Ends",
      "Backend": "One Back Ends",
      "Platform": "x86/ARM/GPU/etc",
      )
    )
  ]

  #only(7)[
    #raw-render(
    ```dot
      digraph {
        Framework -> Frontend
        Frontend -> OPT [label = "LLVM IR"]
        OPT -> Backend [label = "LLVM IR"]
        Backend -> Platform
      }
    ```,
    labels: (
      "languages" : "C/C++/Python/etc",
      "Frontend": "Various Front Ends",
      "Backend": "One Back Ends",
      "Platform": "x86/ARM/GPU/etc",
      )
    )
  ]
]

#slide(
  session: "PyTorch",
  title: "What is Machine Learning Compiler"
)[

]

#slide(
  session: "PyTorch",
  title: "PyTorch Tensors"
)[
  - Tensors are similar to NumPy's ndarrays, with the addition being that Tensors can also be used on a GPU to accelerate computing.
  - PyTorch provides Tensors that can live either on the CPU or the GPU, and accelerates the computation by a huge amount.
  - PyTorch is also an autograd system. Autograd is a PyTorch package for the differentiation for all operations on Tensors.
]
