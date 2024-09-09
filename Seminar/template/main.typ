#import "../../theme/iTalk.typ": *
#import "@preview/algo:0.3.3": algo, i, d, comment, code

// TODO fill all "TODO" with your information

#show: nenu-theme.with(
  short-title: "TODO",
  short-date: "TODO",
  short-author: "Virgil" 
)

#let argmax = math.op("arg max", limits: true)
#let argmin = math.op("arg min", limits: true)

#title-slide(
  title: "TODO",
  authors: (
    name: "凌典",
    email: "virgiling7@gmail.com"
  ),
  logo: image("../template/fig/nenu-logo-title.png", width: 30%),
  institution: "Northeast Normal University",
  date: "TODO"
)
