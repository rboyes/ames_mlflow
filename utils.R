library(purrr)
library(stringr)

setgit <- function(gitconfig = list(user.name = 'Richard Boyes', user.email = 'rboyes@gmail.com')) {
  
  output = purrr::map(names(gitconfig), function(gitname) {
    system(stringr::str_glue("git config --global {gitname} '{gitconfig[[gitname]]}'"))
  })
  return(output)
}