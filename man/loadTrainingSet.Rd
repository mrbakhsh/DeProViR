% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/functions.R
\name{loadTrainingSet}
\alias{loadTrainingSet}
\title{Load Demo Training Set}
\usage{
loadTrainingSet(
  training_dir = system.file("extdata", "training_Set", package = "DeProViR")
)
}
\arguments{
\item{training_dir}{dir containing a training data.frame .csv
Default set to "extdata/training_testSets".}
}
\value{
data.frame
}
\description{
This function loads demo training set.
}
\examples{
dt <- loadTrainingSet()
}
