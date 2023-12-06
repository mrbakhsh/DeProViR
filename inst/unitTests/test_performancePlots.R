test_performancePlots = function() {
   pred_label <- seq(0,1, length.out = 100)
   truth_label <- rep(c(0,1), each = 50)
   perf <- performancePlots(pred_label, truth_label,tpath = tempdir())
   checkTrue(is.list(perf) == TRUE)
}
