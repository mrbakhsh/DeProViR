test_encodeViralSeq = function() {
   embeddings_index <- gloveImport()
   dt <- loadTrainingSet()
   encoded_seq <- encodeViralSeq(dt, embeddings_index)
   checkTrue(nrow(encoded_seq[["embedding_matrix_v"]]) == 20)
}
