#import "../../theme/iTalk.typ": *

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
  title: "A Diving Odyssey through PyTorch",
  authors: (name: "凌典", email: "virgiling7@gmail.com"),
  logo: image("fig/nenu-logo-title.png", width: 30%),
  institution: "Northeast Normal University",
  date: "2023-12-19"
)

#slide(
  session: "PyTorch",
  title: "What is PyTorch"
)[
  #only((beginning: 1, until: 4))[
    == Pytorch is a AI Framework
    #parbreak()
    #v(.2em)
    + PyTorch define a grammer to describe the network
    + Use the grammer to build the network
  ]

  #only((beginning: 2, until: 4))[
    == PyTorch is a #pin(1)Compiler#pin(2)
      #pinit-highlight(1, 2)
      #parbreak()
      #only((beginning: 3, until: 4))[
        #pinit-point-from(
          fill: black, 
          pin-dx: -5pt, 
          pin-dy: 5pt, 
          body-dx: -10em, 
          body-dy: 8pt, 
          offset-dx: -45pt, 
          offset-dy: 40pt,
          2
        )[
          #text(fill: black)[How to understand the \
          compiler here?]
        ]
    ]
    #only(4)[
          #pinit-point-from(fill: red.darken(10%) ,2)[
            #text(fill: red)[
              As there are many programming languages and\ hardware, we may develop a all in one compiler
            ]
          ]
        ]
  ]
]

#slide(
  session: "PyTorch",
  title: "What is Machine Learning Compiler"
)[

  #only((beginning: 1, until: 2))[
    #figure(
      placement: top,
      image("fig/trad-compiler.png", width: 60%),
      caption: [#pin(1)Traditional Compiler#pin(2)]
    )
    #only(1)[
      #pinit-highlight(1, 2)
      #pinit-point-from(2)[
        So many architectures \
        we need a all-in-one \
        compiler
      ]
    ]
  ]

  #only(2)[
    In traditional compiler, we can define an IR to describe the computation process.
    
    What about ML Compiler?
  ]

  #only(3)[
    // #figure(
    //   image("fig/ML-compiler.png", width: 60%),
    //    caption: "ML Compiler"
    // )
      #figure(
        image("fig/ComputationGraph-example.png", width: 80%),
        caption: "Computation Graph"
      )
      Here is a computation graph, which computes $d = (a * w_1 + w_2) * (a * w_1)$
  ]

  #only((beginning: 4, until: 6))[
    #image("fig/ML-compiler.png")
  ]

  #only(4)[
      #simple-arrow(
      fill: red,
      stroke: 0pt,
      start: (50pt, 0pt),
      end: (280pt, -180pt),
      thickness: 2pt,
      arrow-width: 4,
      arrow-height: 4,
      inset: 0.5,
      tail: (),
    )
      #text(fill: red)[Model here can be seen as programming\
      language]
    ]

  #only(5)[
      #simple-arrow(
      fill: red,
      stroke: 0pt,
      start: (100pt, 0pt),
      end: (500pt, -220pt),
      thickness: 2pt,
      arrow-width: 4,
      arrow-height: 4,
      inset: 0.5,
      tail: (),
    )
    #text(fill: red)[a.k.a computation graph]
  ]

  #only(6)[
      #simple-arrow(
      fill: red,
      stroke: 0pt,
      start: (100pt, 0pt),
      end: (550pt, -80pt),
      thickness: 2pt,
      arrow-width: 4,
      arrow-height: 4,
      inset: 0.5,
      tail: (),
    )
    #text(fill: red)[A backend to translate the Graph IR to \
    executable machine code on different device]
  ]
]

#slide(
  session: "PyTorch",
  title: "Structure of PyTorch"
)[
  #only(1)[
    As we know, a compiler can be decomposed into 2 parts:
    + front end
    + back end
    Then what about PyTorch?
  ]
  #only(2)[
    #figure(
      image("fig/pytorch_compiler.drawio.png", height: 90%),
      caption: "PyTorch Structure"
    )
  ]
]

#slide(
  session: "Front End",
  title: "FX Graph"
)[
    #only((until: 1))[
      As we have seen, PyTorch use a graph to describe the computation process, which is called `Computation Graph`

      But how does PyTorch describe the computation graph and how does it caputure the graph?
    ]
    #only(2)[
      Before PyTorch 2.0, PyTorch use FX Graph to represent the computation graph, example code:
    ]
    
    #only(2)[
        #grid(
        columns: (100%),
        rows: (5fr, 10fr)
      )[
          ```python
        class MyModule(nn.Module):
          def __init__():
            super().__init__()
            self.param = \ 
              torch.nn.Parameter(torch.rand(3, 4))
            self.linear = torch.nn.Linear(4, 5)

          def forward(self, x):
            return self.linear( \
            x + self.param \
            ).clamp(min=0.0, max=1.0)
        ```
        #only(2)[
          #set align(right)
          #image("fig/FX_Graph_example.png", fit: "contain")
        ]
      ]
    ]
]

#slide(
  session: "Front End",
  title: "How to capture the graph"
)[
  #only(1)[
      #set align(center + horizon)
      #text(size: 1.5em)[
        We use a front end module called #pin(3) `dynamo` #pin(4) to capture the Computation Graph.

        #pinit-highlight(3, 4)
        ]
      // #image("fig/dynamo_.png", fit: "contain", width: 80%, height: 85%)
      
  ]
  // #only(2)[
  //   #set align(center + horizon)
  //   #text(size: 1.5em)[
  //     Before we diving into it
      
  //     we need to know some basic concepts
  //   ]
  //   + How does Python execute
  //   + What is Frame
  // ]
]

#slide(
  session: "Front End",
  title: "torch._dynamo (high level) intro"
)[

  #only((until: 2))[
      Why called `dynamo`?
  ]

  #only(2)[
    Historical records and official documentation indicate that the name `dynamo` can be traced back to a 1998 technical report by HP, where the term `dynamic optimization` was used.
  ]

  #only((beginning: 3))[
    #v(1em)
    #set align(center + horizon)
    #text(size: 1.5em)[dynamo is a Python-level #strong[Just-in-Time] (JIT) compiler]
    #v(2em)
    #text(size: 1.2em)[Before diving into dynamo, we need to know some basic concepts]
  ]
]

#focus-slide()[
  #set align(center)
  #strong[How does Python execute?]
]


#slide(
  session: "CPython",
  title: "Intro"
)[
  #only(1)[
    Python interpreter is written in C, which is called CPython.

    A Python code runs in CPython is like this:

    #grid(
      columns: (60%, 40%),
      image("fig/CPython-workflow.png", height: 70%, fit: "contain"),
      text[
        + `.py` source code translate into bytecode
        + CPython VM loop and execute the bytecode 
      ]
    )
  ]

  #only(2)[
    A example like this:
    #grid(
      columns: (40%, 10%, 50%),
      text[
        ```py

        def add(x, y):
            return x + y
        ```
      ],
      simple-arrow(
        fill: red,
        start: (-50pt, 50pt),
        end: (50pt, 50pt),
        thickness: 4pt,
        arrow-width: 4,
        arrow-height: 4,
        inset: 0,
        tail: (),
      ),
      text[
        2 #h(1em) LOAD_FAST #h(5em) 0 (x)

        4 #h(1em) LOAD_FAST #h(5em) 1 (y)

        6 #h(1em) BINARY_OP #h(5.2em) 0 (+)

        10 #h(.5em) RETURN_VALUE
      ]
    )
    The bytecode of function `add` is on right

    The VM will execute the bytecode line by line
 
  ]
]


// #slide(
//   session: "CPython",
//   title: "Frame and Frame Evaluation"
// )[
//   #only(1)[
//     A `Frame` is a bit different from `Stack Frame` in C, it is a data structure to store the local variables and some other information.

//     We can describe a `Frame` like this:
//     #grid(
//       columns: (70%, 30%),
//       image("fig/CPython-Frame-example.png", width: 80%, height: 60%, fit: "contain"),
//       text[
//         The call stack of a function is essentially the process of recursively creating #highlight[`Frame`] and executing them.
//       ]
//     )
//   ]

//   #only((2, 3))[
//     #grid(
//       columns: (80%, 20%),
//       text[
//         ```py
//       def foo():
//           frame = inspect.currentframe()
//           cur_func_frame = frame
//           print(f'code name of current frame is {cur_func_name.f_code.co_name}')
//           prev_func_frame = frame.f_back
//           print(f'current code name of previous frame is {prev_func_frame.f_code.co_name}')

//       def bar(a=1):
//           foo()

//       if __name__ == '__main__':
//           bar()
//         ```
//       ],
//       text[
//         #only(2)[#highlight(fill: rgb("#f2a1cc"))[result is: ]]

//         code name of current frame is #highlight[foo] 

//         current code name of previous frame is #highlight[bar]

//         #only(3)[
//           so we can get all previous frame info in current frame
//         ]
//       ]
//     )
//   ]

//   #only((4, 5))[
//     #set align(center + horizon)
//     #text(size: 1.5em)[
//       How about stealing the future?
//     ]

//     #set align(left + horizon)
//     If we know the function frame before we execute it, then we can do some awesome things to the frame, like:
//     #v(.5em)
//     - inject some code into the frame
//     - changing the execution order of the code
//     - etc.
    
//   ]
//   #only(5)[
//     #set align(center + horizon)
//     #text(size: 1.2em)[
//       Python Enhancement Proposal 523 (PEP 523) is a proposal to add a new API to the Python interpreter to allow #strong[stealing the future].
//     ]
//   ]

// ]

// #focus-slide()[
//     #set align(center + horizon)
//     So, what does dynamo do?
// ]

// #slide(
//   session: "Front End",
//   title: "Frame Evaluation"
// )[
//   #only(1)[
//     #set align(center + horizon)
//     #image("fig/PEP-523.png", fit: "stretch")
//   ]

//   #only((2, 3))[
//     #grid(
//       columns: (40%, 60%),
//       column-gutter: 3em,
//       image("fig/PEP-523-dynamo.png", width: 120%, fit: "stretch"),
//       text[
//         `dynamo` change the `_PyEval_EvalFrameDefault` function to its self-defined function 

//         `dynamo` makes a clever choice by performing bytecode parsing at the Python layer and \
//         passing it as a callback function to a frame evaluation function.
        
//         #only(3)[
//           When we invoke `optimizer('inductor')(fn)`, `dynamo` replaces the frame evaluation function \
//           of `fn` with its own custom one, and passes the callback function as an argument.
//         ]
//       ]
//     )
//   ]
// ]

#slide(
  session: "Front End",
  title: "Captures Computation Graph"
)[

  #only(1)[
    #set align(center + horizon)
    #text(size: 1.2em)[
      `dynamo` captures the computation graph by #strong[stealing the future]

      But we will not discuss the details of this, we only care about that `dynamo` captures the computation graph before executing into bytecode.
    ]
  ]

  #only(2)[
    #text(size: 1.2em)[
      Still remember FX Graph?
    ]

    In general, we could simply map each statement to a node in the graph like this:
    #grid(
      columns: 2,
      column-gutter: 3mm,
    text[
      ```py
    def forward(self, x):
      return self.linear(
        x + self.param
      ).clamp(min=0.0, max=1.0)
    ```],
    image("fig/FX_Graph_example.png", fit: "contain")
    )

    But it is not enough, we need to capture the control flow, like `if` statement.
  ]

  #only((3, 4))[
    Let us see a example, when we have a control flow like this:
      ```python
      def forward(self, x):
        if x.sum() < 0:
            return x + 1
        else:
            return x - 1
      ```
      We can not capture the control flow in FX Graph
      
      Because FX Graph is a `symbolic trace`, i.e. it will not do any computation, just record the computation.

    ]

    #only(4)[
      Then what does `dynamo` do?
    ]

    #only((5, 6))[
      Let's see:
      #grid(
       columns: 2,
       column-gutter: 5em,
       text[
        ```python
        def forward(x):
          if x.sum() < 0:
              return x + 1
          else:
              return x - 1
        ```
        ],
        grid(
          columns: 1,
          rows: 2,
          row-gutter: 2em,
          rect[
            #only((5, 6))[
              ```py
              def forward(self, x: torch.Tensor):
                sum_1 = x.sum(); x = None
                lt = sum_1 < 0; sum_1 = None
                return (lt, )
              ```
            ]
          ],
          rect[
            #only((5, 6))[
              ```py
              def forward(self, x: torch.Tensor):
                add = x + 1; x = None
                return (add, )
              ```
            ]
          ]
        )
      )
      #only(6)[
        #text(size: 1.2em)[
          ...wait, Why there are only 2 computation graph?\
          Where is the `else` branch?
        ]
      ]
    ]
]

#slide(
  session: "Front End",
  title: "Guard"
)[

  #only(1)[
    What is Guard?
    
    As we have said, PyTorch is #pin(1)JIT#pin(2) compiler.
    
    #pinit-highlight(1, 2)

    It means PyTorch will #pin(5)compile the code when it is running#pin(6).

    #pinit-highlight(5, 6)

    But we do not want to compile the code every time we run it.
    
    So we need to #pin(3)guard the code#pin(4), dynamically compile the code when it is running for the first time, and cache the compiled code for later use.

    #pinit-highlight(3, 4)

  ]

  #only((beginning: 2, until: 4))[
    #grid(
      columns: (45%, 55%),
      column-gutter: 3mm,
      text[
        ```py
        class MyModule(nn.Module):
          def __init__():
            super().__init__()
            self.linear = torch.nn.Linear(4, 5)

          def forward(self, x):
            return self.linear( \
            x + self.param \
            ).clamp(min=0.0, max=1.0)
        ```
      ],
      [
        #only(2)[
          We need `Guard` to guard the computation graph in case of recompiling the code every time we run it.
        ]
        #only(3)[
          ```py
          local 'self' NN_MODULE
          {
              'guard_types': ['ID_MATCH'],
              'code': ['___check_obj_id(self, 140260021010384)'],
              'obj_weakref': <weakref at 0x7f8f9864a110; to 'Model' at 0x7f90d4ba7fd0>
              'guarded_class': <weakref at 0x7f90d4bc71a0; to 'type' at 0x705e0f0 (Model)>
          }
          ```
        ]
        #only(4)[
          ```py
          local 'x' TENSOR_MATCH
          {
              'guard_types': ['TENSOR_MATCH'],
              'code': None,
              'obj_weakref': <weakref at 0x7f9035c8bce0; to 'Tensor' at 0x7f8f95bddd00>
              'guarded_class': <weakref at 0x7f8f98cf9440; to 'torch._C._TensorMeta' at 0x57f3e10 (Tensor)>
          }
          ```
        ]
      ]
    )
  ]

  #only((beginning: 5, until: 7))[
    #grid(
      columns: (55%, 45%),
      [
        #only((beginning: 5, until: 7))[
          ```py
          local 'x' TENSOR_MATCH
          {
              'guard_types': ['TENSOR_MATCH'],
              'code': None,
              'obj_weakref': <weakref at 0x7f9035c8bce0; to 'Tensor' at 0x7f8f95bddd00>
              'guarded_class': <weakref at 0x7f8f98cf9440; to 'torch._C._TensorMeta' at 0x57f3e10 (Tensor)>
          }
          ```
        ]
      ],
      [
        #only((5, 6))[
          The `Guard` has few ways to check a variable have changed or not.
        ]
        #only(6)[
          + `check_obj_id`: i.e. `ID_MATCH`
          + `check_tensor`: i.e. `TENSOR_MATCH`
        ]
        #only(7)[
          When check `Tensor`, `Guard` will check the `Tensor`'s: 
          1. `dtype`: `fp32` to `fp16`, etc.
          2. `device`: `GPU0` to `GPU1`, etc.
          3. `shape`: `(1, 2)` to `(2, 1)`, etc.
          4. `requires_grad`: `True` to `False`, etc.
          5. `alignment`: `32` to `64`, etc.

          It will re-compile when all check failed.
        ]
      ]
    )
  ]
]

#slide(
  session: "Front End",
  title: "Graph Break"
)[

  #only(1)[
    #grid(
      columns: 2,
      column-gutter: 10mm,
      text[
      ```python
      def forward(self, x):
        if x.sum() < 0:
            return x + 1
        else:
            return x - 1
      ```
    ],
    text[
      With `Guard`, `dynamo` can guarantee that if the tensor does not change, the computation graph will not change, so we do not need to compiler the `else` branch.


      But... what if the tensor changes?
    ]
    ),
  ]

  #only(2)[
    let's see this example:
  ]

  #only((beginning:2, until: 7))[
    #grid(
     columns: 2,
    column-gutter: 12mm,
     [
      #only((2, 3))[
        ```py
        def toy_example(x):
          a = nn.Linear(1, 1)(x)
          b = nn.Linear(1, 1)(x)
          if x.sum() < 0:
              return a + b
          return a - b
        ```
      ]
      #only((4, 5, 6, 7))[
        ```py
        def compiled_toy_example(x):
          a, b, lt = __compiled_fn_0(x)
          if lt:
              return __resume_at_30_1(b, x)
          else:
              return __resume_at_38_2(a, x)
        ```
      ]
     ],
     [
      #only(2)[
        `dynamo` here take an action called `Graph Break`

        `toy_example()` would be compiled into 3 computation graph.
      ]
      #only(3)[
        ```py
        def compiled_toy_example(x):
          a, b, lt = __compiled_fn_0(x)
          if lt:
              return __resume_at_30_1(b, x)
          else:
              return __resume_at_38_2(a, x)
        ```
        three function shows below:
      ]
      #only(4)[
        before `if` statement:

        ```py
        def __compiled_fn_0(x):
          a, b = nn.Linear(1, 1)(x), nn.Linear(1, 1)(x)
          return x.sum() < 0:
        ```
      ]
      #only(5)[
        `if` branch:

        ```py
        def __resume_at_30_1(x):
          goto if_next
          a, b = nn.Linear(1, 1)(x), nn.Linear(1, 1)(x)
          if x.sum() < 0:
              label if_next
              return a + b
          return a - b
        ```
      ]
      #only(6)[
        `else` branch:

        ```py
        def __resume_at_38_2(x):
          goto if_jump
          a, b = nn.Linear(1, 1)(x), nn.Linear(1, 1)(x)
          if x.sum() < 0:
              b = a + b
          label if_jump
          return a - b
        ```
      ]
      #only(7)[
        At first, the function `__compiled_fn()` will be compiled, and we have the `lt` flag

        Then `dynamo` will compile the `if` branch #pin(1)*or*#pin(2) `else` branch according to the `lt` flag.

        #pinit-highlight(1, 2)
      ]
     ]
    )
  ]
]

#slide(
  session: "Front End",
  title: "Loop Unrolling"
)[

  #only(1)[
    #set align(center + horizon)
    #text(size: 1.2em)[
       The control flow also includes `for` and `while` statement, how does `dynamo` handle them?
    ]
  ]

  #only((2, 3))[
    #only(2)[Let's see an example:]
    #grid(
     columns: 2,
     column-gutter: 20mm,
     [
      ```py
      @torch.compile
      def toy_example(x, n):
          for i in range(1, n + 1):
              x = x * i
          return x

      def test():
          x = torch.randn(10)
          toy_example(x, 4)
      ```
     ],
     [
      #only(3)[
        The Graph of `toy_example` is like this:
        ```py
        def forward(self, x : torch.Tensor):
          mul = x * 1;  x = None
          mul_1 = mul * 2;  mul = None
          mul_2 = mul_1 * 3;  mul_1 = None
          mul_3 = mul_2 * 4;  mul_2 = None
          return (mul_3,)
        ```
      ]
     ]
    )
  ]
]

#slide(
  session: "Front End",
  title: "dynamo"
)[
  We can summary dynamo like this:
  #image("fig/dynamo_.png", fit: "contain", width: 100%, height: 85%)
]

#slide(
  session: "Front End",
  title: "AOTAutograd"
)[

  

]
