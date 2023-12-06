test_loadPreTrainedModel = function() {
   trained_model <- loadPreTrainedModel()
   checkTrue(typeof(trained_model) == "closure")
}
