#import "../../theme/iTalk.typ": *
#import "@preview/algo:0.3.3": algo, i, d, comment, code

// TODO fill all "TODO" with your information

#show: nenu-theme.with(
  short-title: "SparrOS",
  short-date: "2024-05-08",
  short-author: "凌典" 
)

#let argmax = math.op("arg max", limits: true)
#let argmin = math.op("arg min", limits: true)

#title-slide(
  title: "SparrOS",
  subtitle: "面向 RISC-V 的 POSIX 操作系统内核设计与实现",
  authors: (
    (
      name: "答辩学生：凌典",
      email: "virgiling7@gmail.com"
    ),
    (
      name: "指导老师：胡书丽",
      email: "husl903@nenu.edu,cn"
    )
  ),
  logo: image("../../Seminar/template/fig/nenu-logo-title.png", width: 30%),
  institution: "Northeast Normal University",
  date: "2024-05-08"
)

#toc-slide()

#slide(
  session: "选题背景",
  title: "选题背景"
)[
  // TODO 选题背景主要从操作系统的教学入手，主要说明两点：
  //      1. 国内操作系统教学的问题所在
  //      2. 工业级别操作系统的复杂性 
  纵观计算机系统的历史，已经有过太多操作系统的出现，
  
  - IBM 的 OS/360
  - MIT 的 CTSS
  - Unix#footnote[Ritchie, Dennis M. and Ken Thompson. “The UNIX time-sharing system.” The Bell System Technical Journal 57 (1974): 1905-1929.]，Minix#footnote[Tanenbaum, Andrew S.. “Operating systems: design and implementation.” Prentice-Hall software series (1987)]
  - Linux#footnote[Linux, https://kernel.org/]，Windows，MacOS，Open Harmony#footnote[⁴OpenHarmony: https://www.openharmony.cn/mainPlay]
这些操作系统的架构各有差异，但都是工业级别的操作系统，由于需要兼容各
类硬件，其代码行数高达十几万行，对于初学者而言难度太大。
]

#slide(
  title: "选题背景"
)[
对于教学级别的操作系统而言，例如
- MIT 的 xv6#footnote[xv6: https://pdos.csail.mit.edu/6.S081/2023/xv6/book-riscv-rev3.pdf]
- 清华大学的 uCore, rCore
- UCB 的 PintOS#footnote[Pfaff, Ben, Anthony Romano and Godmar Back. “The pintos instructional operating system kernel.” Technical Symposium on Computer Science Education
(2009).]
其实现均为 C 语言，但 C 语言需要谨慎和经验才能安全使用，即便如此，低级错误也屡见不鲜。

而南京大学的 Mosaic#footnote[Jiang, Yanyan. “The Hitchhiker’s Guide to Operating Systems.” USENIX Annual Technical Conference (2023).]，其只关注了操作系统的抽象，而屏蔽了底层的太多细节

]


// #slide(
//   session: "论文综述",
//   title: "论文综述"
// )[

// ]

#slide(
  session: "设计与实现",
  title: "研究目标"
)[
  本文旨在提出一种由高级语言重构的 POSIX 操作系统内核，此内核面向简洁的 RISC-V 架构，且能够兼容部分 POSIX.1 标准。
]

#slide(
  title: "总体架构"
)[
  #figure(
    image("fig/sparros-arch.png"),
  )
]

#slide(
  title: "虚拟化"
)[
  = 内存虚拟化
  = CPU 虚拟化
  = 文件系统虚拟化
]

#slide(
  title: "内存虚拟化"
)[
  我们通过 sv39 页表机制来实现内存虚拟化，其转化机制如下图所示：
  #set align(center)
  #image("fig/sv39-pte.png", fit: "contain", height: 80%)
]

#slide(
  title: "内存虚拟化"
)[
  并通过 TLSF 算法来进行空闲内存的管理与分配，其管理的数据结构如下：
  #set align(center)
  #image("fig/tlsf-demo.png", fit: "contain", height: 80%)
]

#slide(
  title: "CPU 虚拟化"
)[

  CPU 虚拟化的核心在于对进程的抽象，在 SparrOS 中，我们定义进程如下：

  #code()[
    ```rust
 pub struct Proc {
    pub kstack: KVAddr, // Virtual address of kernel stack
    pub sz: usize, // Size of process memory (bytes)
    pub uvm: Option<Uvm>, // User Memory Page Table
    pub context: Context, // swtch() here to run process
    pub pid: PId, // Process ID
    pub state: ProcState, // Process state
    pub VRunTime: usize, // Virtual running time
    // ...
 }
    ```
  ]
]

#slide(
  title: "CPU 虚拟化"
)[
  我们通过 CFS 来实现进程的调度，以实现 CPU 的虚拟化，我们为每一个 CPU 都建立了一个进程队列 `rq`，其结构如下图所示：
  #set align(center)
  #image("fig/cfs-demo.png", fit: "contain", height: 75%)
]

#slide(
  title: "持久化"
)[
  SparrOS 的文件系统实现参考了 rCore 的 easy-fs 与 xv6 的文件系统，我们将磁盘分为以下几个部分：
  #figure(
    image("fig/easy-fs.png"),
  )
]

#slide(
  title: "文件系统虚拟化"
)[
  我们通过上述的 `inode` 层进行虚拟化，将对文件的操作封装为统一的接口，例如 `open`, `write`, `read` 等系统调用。
]

#slide(
  title: "设备管理"
)[
  我们通过设备树进行管理，在 QEMU 中模拟的 virt 设备，其设备树部分如下：
  #figure(
    image("fig/device_tree.png", fit: "contain", height: 80%),  
  )
]

#slide(
  title: "设备管理"
)[
  我们实现了 `virtio` ，`rtc` 与 `serial` 的驱动程序，`rtc` 的驱动程序如下所示：

  #code()[
    ```rs
impl RTCDriver for GoldfishRTC {
  fn get_time(&self) -> TimeStramp {
    unsafe {
      let time_low = read_volatile((self.base + TIMER_TIME_LOW as usize) as *const u32);
      let time_high = read_volatile((self.base + TIMER_TIME_HIGH as usize) as *const u32);
      let time = ((time_high as u64) << 32) | (time_low as u64);
      TimeStramp::new(sec, NSEC_PER_SEC)
    }
  }
}
    ```
  ]
]

#slide(
  title: "并发"
)[
  在并发方面，我们主要聚焦于实现进程的同步与互斥问题，SparrOS 中实现了包括：
  + `spinlock`
  + `sleeplock`
  + `condvar`
]

#slide(
  title: "并发"
)[
  通过条件变量实现了“生产者与消费者”问题（场景为 `UART` 驱动），如下所示：
  
  #figure(
    image("fig/buffer-limit.png", fit: "contain", height: 80%)
  )
]

#slide(
  session: "实验与结果",
  title: "实验环境"
)[

#set align(center + horizon)
#figure(
  table(
    columns: 3,
    stroke: (x: none),
    row-gutter: (2.2pt, auto),
  table.header[环境条目][宿主机配置][版本],
[操作系统], [Manjaro-Linux], [6.1.84-1],
[CPU 架构], [Intel x86-64], [i5-13500H],
[虚拟机版本], [QEMU], [8.2.2],
[代码编辑器], [VS Code], [1.88.1],
[调试工具], [gdb-multiarch], [12.1],
[交叉编译工具 (1)], [Rust], [nightly-1.79.0],
[交叉编译工具 (2)], [LLVM], [18.1.2],
[运行内存], [16G], [],
[磁盘空间], [NVMe], [512G],
  ),
)
]


#slide(
  title: "实验结果"
)[
  我们通过编写用户程序并写入文件系统来测试内核的正确性：
  + shell
  + ls (list)
  + 并行素数筛
]

#slide(
  title: "并行素数筛"
)[
  这里我们展示素数筛的工作流程及其实验结果，其主要使用 `pipe`，将多个进程连接为一个流水线，从而达到并行的效果，效果图如下所示：

  #figure(
    image("fig/sieve.gif")
  )
]

#slide(
  title: "并行素数筛"
)[
  测试结果如下图所示：
  #figure(
    image("fig/primer.png", fit: "contain", height: 80%),
  )
]


#slide(
  session: "总结与展望",
  title: "工作总结"
)[
  + 设计并实现了一个兼容部分 POSIX.1 标准的操作系统。
  + 设计各类模块时注重了现代化这一特点，采用了 TLSF 分配算法，CFS 调度算法以及设备树等现代操作系统使用的特性。
  + 使用了 Rust 这种高级程序语言进行开发，虽然损失了一定的程序性能，但高级语言的抽象可以极大提高程序的可读性与可维护性。
]

#slide(
  title: "未来展望"
)[
  + 在完成了操作系统基本功能的基础上，继续完善网络模块，例如增加 TCP/IP 协议栈。
  + 将文件系统升级，改为ext4 文件系统从而能够与 Unix-like 的 OS 兼容，并增加更多的虚拟文件系统，例如 procfs 与 devfs。
  + 将操作系统烧录到开发版上运行，并实现开发板的设备驱动。
  + 升级操作系统支持的硬件架构，例如 LoongArch 等 RISC 指令集架构。
]

#focus-slide[
  #set align(center + horizon)
  Q & A
]

#focus-slide[
  #set align(center + horizon)
  恳请老师批评指正
]
