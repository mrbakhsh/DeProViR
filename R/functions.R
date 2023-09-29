   #' gloveEmb_import
   #' @title Cache and Load Pre-Trained Word Vectors
   #' @param url_path  URL path to GloVe embedding. Defaults to
   #' "https://nlp.stanford.edu/data"
   #' @return glove embedding
   #' @description This function cache and loads pre-trained GloVe
   #' vectors (100d).
   #' @export
   #' @importFrom BiocFileCache BiocFileCache
   #' @importFrom BiocFileCache bfcrpath
   #' @importFrom BiocFileCache bfcinfo
   #' @importFrom BiocFileCache bfcquery
   #' @importFrom utils unzip
   #' @examples
   #' options(timeout=240)
   #' embeddings_index <-
   #' gloveEmb_import(url_path = "https://nlp.stanford.edu/data")



    gloveEmb_import <-
        function(url_path = "https://nlp.stanford.edu/data") {


           url <- paste(
              url_path,
              "glove.6B.zip",
              sep="/")
            bfc <- BiocFileCache()
            path <- bfcrpath(bfc, url)
            getid <- bfcquery(bfc, "glove")$rid
            lines <- readLines(unzip(zipfile = bfcinfo(bfc[getid])$rpath,
                                     files = "glove.6B.100d.txt",
                                     exdir = tempfile()))
            embeddings_index <- new.env(hash = TRUE, parent = emptyenv())
            for (i in seq_along(lines)) {
                line <- lines[[i]]
                values <- strsplit(line, " ")[[1]]
                word <- values[[1]]
                embeddings_index[[word]] <- as.double(values[-1])
            }

            return(embeddings_index)

        }



   #' load_TrainingSet
   #' @title Load Demo Training Set
   #' @param training_dir dir containing a training data.frame .csv
   #' Default set to "extdata/training_testSets".
   #' @return data.frame
   #' @description This function loads demo training set.
   #' @importFrom readr read_csv
   #' @importFrom readr col_double
   #' @importFrom readr col_character
   #' @importFrom readr cols
   #' @export
   #' @examples
   #' dt <- load_TrainingSet()

   load_TrainingSet <-
      function(training_dir =  system.file("extdata", "training_Set",
                                           package = "DeProViR")) {



         csv_files <- list.files(training_dir, pattern = "\\.csv$",
                                 full.names = TRUE)

         path <-  file.path(csv_files)

         col_types <- cols(
            Human_ID = col_character(),
            Virus_ID = col_character(),
            Human_Seq = col_character(),
            Virus_Seq = col_character(),
            Label = col_double()
         )

         dt <- read_csv(path, col_types = col_types)
         return(dt)
      }


   #' encode_ViralSeq
   #' @title Viral Protein Sequence Encoding with GloVe Embedding Vectors
   #' @param trainingSet a data.frame containing training information
   #' @param embeddings_index embedding outputted from
   #' \code{\link[DeProViR]{gloveEmb_import}}
   #' @return A list containing Embedding matrix and tokenization
   #' @description This function first first encodes amino acids as a sequence
   #' of unique 20 integers though tokenizer. The padding token was added to the
   #' front of shorter sequences to ensure a fixed-length vector of
   #' defined size L (i.e., here is 1000). Embedding matrix is then constructed
   #' to transform amino acid tokens to pre-training embedding weights, in which
   #' rows represent the amino acid tokens created earlier, and columns
   #' correspond to 100-dimension weight vectors derived from GloVe
   #' word-vector-generationvector map.
   #' @importFrom keras text_tokenizer
   #' @importFrom keras fit_text_tokenizer
   #' @importFrom keras texts_to_sequences
   #' @importFrom keras pad_sequences
   #' @importFrom dplyr %>%
   #' @export
   #' @examples
   #' # Download and load the index
   #' embeddings_index <- gloveEmb_import()
   #' #load training set
   #' dt <- load_TrainingSet()
   #' #encoding
   #' encoded_seq <- encode_ViralSeq(dt, embeddings_index)

   encode_ViralSeq <- function(trainingSet,
                               embeddings_index) {

      v_text <- strsplit(trainingSet$Virus_Seq, "")
      v_text <- unname(rapply(v_text, paste, collapse=" "))
      # tokenize the data
      max_words <- 20 # Considering only the standard amino acid
      tokenizer <- text_tokenizer(num_words = max_words) %>%
         fit_text_tokenizer(v_text)
      v_sequences <- texts_to_sequences(tokenizer, v_text)
      maxlen <- 1000                 # We will cut texts after 1000 words
      data_v <- pad_sequences(v_sequences, maxlen = maxlen)
      data_v <- as.matrix(data_v)
      row.names(data_v) <- trainingSet$Virus_ID


      word_index <- tokenizer$word_index
      embedding_dim <- 100
      embedding_matrix_v <- array(0, c(max_words, embedding_dim))
      for (word in names(word_index)) {
         index <- word_index[[word]]
         if (index < max_words) {
            embedding_vector <- embeddings_index[[word]]
            if (!is.null(embedding_vector))
               # Words not found in the embedding index will be all zeros.
               embedding_matrix_v[index+1,] <- embedding_vector
         }
      }
      output <- list()
      output$embedding_matrix_v <- embedding_matrix_v
      output$data_v <- data_v
      return(output)

   }


   #' encode_HostSeq
   #' @title Host Protein Sequence Encoding with GloVe Embedding Vectors
   #' @param trainingSet a data.frame containing training information
   #' @param embeddings_index embedding outputted from
   #' \code{\link[DeProViR]{gloveEmb_import}}
   #' @return A list containing Embedding matrix and tokenization
   #' @description This function first first encodes amino acids as a sequence
   #' of unique 20 integers though tokenizer. The padding token was added to the
   #' front of shorter sequences to ensure a fixed-length vector of
   #' defined size L (i.e., here is 1000). Embedding matrix is then constructed
   #' to transform amino acid tokens to pre-training embedding weights, in which
   #' rows represent the amino acid tokens created earlier, and columns
   #' correspond to 100-dimension weight vectors derived from GloVe
   #' word-vector-generation vector map.
   #' @export
   #' @examples
   #' # Download and load the index
   #' embeddings_index <- gloveEmb_import()
   #' #load training set
   #' dt <- load_TrainingSet()
   #' #encoding
   #' encoded_seq <- encode_HostSeq(dt, embeddings_index)


   encode_HostSeq <- function(trainingSet,embeddings_index) {

      h_text <- strsplit(trainingSet$Human_Seq, "")
      h_text <- unname(rapply(h_text, paste, collapse=" "))
      # tokenize the data
      max_words <- 20    # Considering only the standard amino acid
      tokenizer <- text_tokenizer(num_words = max_words) %>%
         fit_text_tokenizer(h_text)
      h_sequences <- texts_to_sequences(tokenizer, h_text)
      maxlen <- 1000                 # We will cut texts after 1000 words
      data_h <- pad_sequences(h_sequences, maxlen = maxlen)
      data_h <- as.matrix(data_h)
      row.names(data_h) <- trainingSet$Human_ID

      word_index <- tokenizer$word_index
      embedding_dim <- 100
      embedding_matrix_h <- array(0, c(max_words, embedding_dim))
      for (word in names(word_index)) {
         index <- word_index[[word]]
         if (index < max_words) {
            embedding_vector <- embeddings_index[[word]]
            if (!is.null(embedding_vector))
               # Words not found in the embedding index will be all zeros.
               embedding_matrix_h[index+1,] <- embedding_vector
         }
      }
      output <- list()
      output$embedding_matrix_h <- embedding_matrix_h
      output$data_h <- data_h
      return(output)
   }



   #' ModelPerformance_evalPlots
   #' @title Model Performance Evalution
   #' @param pred_label predicted labels
   #' @param y_label Ground truth labels
   #' @param tpath A character string indicating the path to the project
   #' directory. If the directory is
   #' missing, PDF file will be stored in the Temp directory.
   #' @return Pdf file containing perfromanc plots
   #' @description This function plots model performance
   #' @importFrom caret confusionMatrix
   #' @importFrom fmsb radarchart
   #' @importFrom pROC roc
   #' @importFrom PRROC pr.curve
   #' @importFrom pROC ggroc
   #' @importFrom ggplot2 aes
   #' @importFrom ggplot2 ggplot
   #' @importFrom ggplot2 annotate
   #' @importFrom ggplot2 coord_equal
   #' @importFrom ggplot2 element_blank
   #' @importFrom ggplot2 element_line
   #' @importFrom ggplot2 element_text
   #' @importFrom ggplot2 geom_abline
   #' @importFrom ggplot2 scale_y_continuous
   #' @importFrom ggplot2 theme
   #' @importFrom ggplot2 theme_bw
   #' @importFrom ggplot2 unit
   #' @importFrom ggplot2 xlab
   #' @importFrom ggplot2 geom_line
   #' @importFrom ggplot2 ylab
   #' @importFrom grDevices dev.off
   #' @importFrom grDevices pdf
   #' @importFrom grDevices recordPlot
   #' @export
   #' @examples
   #' pred_label = seq(0,1, length.out = 100)
   #' truth_label = rep(c(0,1), each = 50)
   #' perf <- ModelPerformance_evalPlots(pred_label, truth_label,
   #' tpath = tempdir())


   ModelPerformance_evalPlots <-
      function(pred_label, y_label, tpath = tempdir()) {

         V1 <- NULL
         V2 <- NULL

         pdf(file.path(tpath, "plots.pdf"))
         # confusion matrix
         pred <-
            as.factor(ifelse(pred_label > 0.5, "1", "0"))

         cm <-
            confusionMatrix(data=pred,
                            reference = as.factor(y_label), positive = "1")

         # Generate plot for cm result
         df <-
            data.frame(rbind(rep(1,6),
                             rep(0,6), cbind(cm$byClass[["Sensitivity"]],
                                             cm$byClass[["Specificity"]],
                                             cm$byClass[["Precision"]],
                                             cm$byClass[["Recall"]],
                                             cm$byClass[["F1"]],
                                             cm$byClass[["Balanced Accuracy"]]
                                             )))
         colnames(df) <- c("SE","SP","PPV","Recall","F1", "Acc")

         raderplot <-
            radarchart(df,cglty = 2, pfcol = c("#99999980"),
                       cglcol = "blue",pcol = 2,plwd = 2, plty = 1)

         p <- recordPlot()


         ## roc curve
         roc_c <-
            pROC::roc(as.numeric(y_label),as.numeric(pred_label))
         roc_plot <-
            pROC::ggroc(roc_c, legacy.axes = TRUE) +
            geom_abline(
               slope = 1, intercept = 0,
               linetype = "dashed", alpha = 0.7, color = "darkgreen"
            ) + coord_equal() + theme_bw() +
            theme(
               axis.line = element_line(colour = "black"),
               panel.grid.major = element_blank(),
               panel.grid.minor = element_blank()
            ) +
            theme(axis.ticks.length = unit(.5, "cm")) +
            theme(text = element_text(size = 14, color = "black")) +
            theme(axis.text = element_text(size = 12, color = "black")) +
            xlab("False Positive Rate (1-Specificity)") +
            ylab("True Positive Rate (Sensitivity)") +
            annotate("text", x=0.25, y=0.8,
                     label= paste("AUC:",round(roc_c[["auc"]],2)))


         # pr analysis
         pr_c <-
            pr.curve(
               scores.class0 = pred_label[y_label == 1],
               scores.class1 =  pred_label[y_label == 0],
               curve = TRUE)

         PR_Object <- as.data.frame(pr_c$curve)

         pr_plot <-
            ggplot(PR_Object, aes(x = V1, y = V2)) +
            geom_line() +
            theme_bw() + scale_y_continuous(limits = c(0, 1)) +
            theme(plot.title = element_text(hjust = 0.5)) +
            theme(
               axis.line = element_line(colour = "black"),
               panel.grid.major = element_blank(),
               panel.grid.minor = element_blank()) +
            theme(axis.ticks.length = unit(.5, "cm")) +
            theme(text = element_text(size = 14, color = "black")) +
            theme(axis.text = element_text(size = 12, color = "black")) +
            xlab("Recall") + ylab("Percision") +
            annotate("text", x=0.25, y=0.25,
                     label= paste("AUC:",round(pr_c[["auc.integral"]],2)))


         plot_list <- list()
         plot_list$roc <- roc_plot
         plot_list$pr <- pr_plot
         plot_list$rader <- p


         for (i in seq_along(plot_list)) {
            print(plot_list[[i]])
         }
         dev.off()

         output <- list()
         output$cm <- cm
         output$roc_c <- roc_c
         output$pr_c <- pr_c

         return(output)


      }


   #' ModelTraining
   #' @title Predictive Model Training using k-fold Validation Strategy
   #' @param url_path  URL path to GloVe embedding. Defaults to
   #' "https://nlp.stanford.edu/data/glove.6B.zip".
   #' @param training_dir dir containing viral-host training set.
   #' See \code{\link[DeProViR]{load_TrainingSet}}
   #' @param input_dim Integer. Size of the vocabulary, i.e. amino acid
   #' tokens. Defults to 20. See \code{keras}.
   #' @param output_dim Integer. Dimension of the dense embedding,
   #' i.e., GloVe. Defaults to 100. See \code{keras}.
   #' @param filters_layer1CNN Integer, the dimensionality of the output space
   #' (i.e. the number of output filters in the first convolution).
   #' Defaults to 32. See \code{keras}
   #' @param kernel_size_layer1CNN An integer or tuple/list of 2 integers,
   #' specifying the height and width of the convolution window in the first
   #' layer. Can be a single integer to specify the same value for all
   #' spatial dimensions.Defaults to 16. See \code{keras}
   #' @param filters_layer2CNN Integer, the dimensionality of the output space
   #' (i.e. the number of output filters in the second convolution).
   #' Defaults to 64. See \code{keras}
   #' @param kernel_size_layer2CNN An integer or tuple/list of 2 integers,
   #' specifying the
   #' height and width of the convolution window in the second layer. Can be a
   #' single integer to specify the same value for all spatial dimensions.
   #' Defaults to 7. See \code{keras}
   #' @param pool_size Down samples the input representation by taking the
   #' maximum value over a spatial window of size pool_size.
   #' Defaults to 30.See \code{keras}
   #' @param layer_lstm Number of units in the Bi-LSTM layer. Defaults to 64.
   #' See \code{keras}
   #' @param units Number of units in the MLP layer. Defaults to 8.
   #' See \code{keras}
   #' @param metrics Vector of metric names to be evaluated by the model
   #' during training and testing. Defaults to "AUC". See \code{keras}
   #' @param cv_fold Number of partitions for cross-validation. Defaults to 10.
   #' @param epochs Number of epochs to train the model. Defaults to 100.
   #' See \code{keras}
   #' @param batch_size Number of samples per gradient update.Defults to 128.
   #' See \code{keras}
   #' @param plots PDF file containing perfromance measures. Defaults to TRUE.
   #' See \code{\link[DeProViR]{ModelPerformance_evalPlots}}
   #' @param tpath A character string indicating the path to the project
   #' directory. If the directory is missing, PDF file containing perfromance
   #' measures will be stored in the Temp directory.
   #' See \code{\link[DeProViR]{ModelPerformance_evalPlots}}
   #' @param save_model_weights If TRUE, save the trained weights.
   #' Defaults to TRUE.
   #' @param filepath A character string indicating the path to save
   #' the model weights. Default to tempdir().
   #' @return Trained model and perfromance measures.
   #' @description  This function first transforms protein sequences to
   #' amino acid tokens wherein tokens are indexed by positive integers,
   #' then represents each amino acid token by pre-trained co-occurrence
   #' embedding vectors learned by GloVe,
   #' followed by applying an embedding layer. Then it employs Siamese-like
   #' neural network articheture on densly-connected neural net to
   #' predict interactions between host and viral proteins.
   #' @importFrom keras layer_input
   #' @importFrom keras layer_embedding
   #' @importFrom keras layer_conv_1d
   #' @importFrom keras layer_max_pooling_1d
   #' @importFrom keras bidirectional
   #' @importFrom keras layer_dropout
   #' @importFrom keras layer_concatenate
   #' @importFrom keras layer_dense
   #' @importFrom keras compile
   #' @importFrom keras keras_model
   #' @importFrom keras fit
   #' @importFrom keras callback_early_stopping
   #' @importFrom keras save_model_weights_hdf5
   #' @importFrom stats predict
   #' @export

   ModelTraining <- function(
      url_path = "https://nlp.stanford.edu/data",
      training_dir = system.file("extdata", "training_Set",
                                 package = "DeProViR"),
      input_dim = 20,
      output_dim = 100,
      filters_layer1CNN = 32,
      kernel_size_layer1CNN = 16,
      filters_layer2CNN = 64,
      kernel_size_layer2CNN = 7,
      pool_size = 30,
      layer_lstm = 64,
      units = 8,
      metrics = "AUC",
      cv_fold = 10,
      epochs = 100,
      batch_size = 128,
      plots = TRUE,
      tpath = tempdir(),
      save_model_weights = TRUE,
      filepath = tempdir()) {


      #### Glove importing
      embeddings_index <-
         gloveEmb_import(url_path)

      message("GLoVe importing is done ....")

      #### Load training set
      dt <-
         load_TrainingSet(training_dir)

      message("Training Set importing is done ....")

      #### viral embedding
      viral_embedding <-
         encode_ViralSeq(dt, embeddings_index)

      message("Viral Embedding is done ....")


      #### host embedding
      host_embedding <-
         encode_HostSeq(dt, embeddings_index)

      message("Hose Embedding is done ....")



      #### network architecture

      seq1 <-
         layer_input(shape = ncol(viral_embedding$data_v), name = "viral_seq")


      m1 <-
         seq1 %>%
         layer_embedding(input_dim = input_dim,
                         output_dim = output_dim,
                         weights = list(viral_embedding$embedding_matrix_v),
                         trainable = FALSE,
                         input_length = ncol(viral_embedding$data_v)) %>%
         layer_conv_1d(filters = filters_layer1CNN,
                       kernel_size =kernel_size_layer1CNN,
                       strides = 2,
                       activation = "relu") %>%
         layer_conv_1d(filters = filters_layer2CNN,
                       kernel_size =kernel_size_layer2CNN,
                       strides = 2,
                       activation = "relu") %>%
         layer_max_pooling_1d(pool_size = pool_size) %>%
         bidirectional(layer_lstm(units = layer_lstm)) %>%
         layer_dropout(rate = 0.3)

      seq2 <-
         layer_input(shape = ncol(host_embedding$data_h), name = "host_seq")

      m2 <-
         seq2 %>%
         layer_embedding(input_dim = input_dim,
                         output_dim = output_dim,
                         weights = list(host_embedding$embedding_matrix_h),
                         trainable = FALSE,
                         input_length = ncol(host_embedding$data_h)) %>%
         layer_conv_1d(filters = filters_layer1CNN,
                       kernel_size =kernel_size_layer1CNN,
                       strides = 2,
                       activation = "relu") %>%
         layer_conv_1d(filters = filters_layer2CNN,
                       kernel_size =kernel_size_layer2CNN,
                       strides = 2,
                       activation = "relu") %>%
         layer_max_pooling_1d(pool_size = pool_size) %>%
         bidirectional(layer_lstm(units = layer_lstm)) %>%
         layer_dropout(rate = 0.3)

      merge_vector <-
         layer_concatenate(list(m1,m2))
      out <-
         layer_dense(units = units, activation = "relu")(merge_vector) %>%
         layer_dense(units = 1, activation = "sigmoid")
      merge_model <-
         keras_model(inputs=list(seq1,seq2),outputs=out)
      merge_model %>% compile(
         optimizer = "adam",
         loss = "binary_crossentropy",
         metrics = metrics
      )

      #### training using k-fold
      data_v <-
         viral_embedding$data_v
      data_h <-
         host_embedding$data_h
      x_train <-
         as.matrix(cbind(data_v,data_h))
      cv_fold =
         cv_fold
      xtr <-
         x_train
      ytr <-
         dt$Label
      prob =
         matrix(nrow = length(ytr), ncol = 1)
      index =
         rep(1:cv_fold, nrow(xtr))
      ind =
         index[1:nrow(xtr)]


      for (k in 1:cv_fold) {
         cat(".")

         # training data
         xcal <- xtr[ind != k, ]
         ycal <- ytr[ind != k]
         xtest <- xtr[ind == k, ]

         history <-
            merge_model %>% fit(
               x =  list(xcal[,1:1000],xcal[,1001:2000]),
               ycal,
               epochs = 100,
               batch_size = 128,
               callbacks =
                  list(callback_early_stopping(monitor = "auc", patience = 3,
                                               restore_best_weights = TRUE)))


         prob[ind == k, ] =
            merge_model %>% predict(list(xtest[,1:1000], xtest[,1001:2000]))
      }
      # model performance evaluation

      if(plots) {

         model_perf <-
            ModelPerformance_evalPlots(prob, dt$Label, tpath)

      }

      if(save_model_weights) {

         filepath = file.path(filepath, "pre_trained_model.h5")
         save_model_weights_hdf5(merge_model, filepath)

      }

      #output <- list()
      #output$model_weights <- merge_model
      #output$model_perf <- model_perf

      return(merge_model)
   }



   #' Load_PreTrainedModel
   #' @title Load Pre-Trained Model Wights
   #' @param input_dim Integer. Size of the vocabulary, i.e. amino acid tokens.
   #' Defults to 20. See \code{keras}.
   #' @param output_dim Integer. Dimension of the dense embedding, i.e., GloVe.
   #' Defaults to 100. See \code{keras}.
   #' @param output_dim Integer. Dimension of the dense embedding, i.e., GloVe.
   #' Defaults to 100. See \code{keras}.
   #' @param filters_layer1CNN Integer, the dimensionality of the output space
   #' (i.e. the number of output filters in the first convolution).
   #' Defaults to 32. See \code{keras}
   #' @param kernel_size_layer1CNN An integer or tuple/list of 2 integers,
   #' specifying the height and width of the convolution window in the first
   #' layer. Can be a single integer to
   #' specify the same value for all spatial dimensions.
   #' Defaults to 16. See \code{keras}
   #' @param filters_layer2CNN Integer, the dimensionality of the output space
   #' (i.e. the number of output filters in the second convolution).
   #' Defaults to 64. See \code{keras}
   #' @param kernel_size_layer2CNN An integer or tuple/list of 2 integers,
   #' specifying the height and width of the convolution window in the
   #' second layer. Can be a single integer to
   #' specify the same value for all spatial dimensions.
   #' Defaults to 7. See \code{keras}
   #' @param pool_size Down samples the input representation by taking the
   #' maximum value over a spatial window of size pool_size.
   #' Defaults to 30.See \code{keras}
   #' @param layer_lstm Number of units in the Bi-LSTM layer. Defaults to 64.
   #' See \code{keras}
   #' @param units Number of units in the MLP layer. Defaults to 8.
   #' See \code{keras}
   #' @param metrics Vector of metric names to be evaluated by the model during
   #' training and testing. Defaults to "AUC". See \code{keras}
   #' @param filepath A character string indicating the path contained
   #' pre-trained model weights, i.e., inst/extdata/Pre-trainedModel
   #' @return Pre-trained model.
   #' @description This function loads the pre-trained model weights constructed
   #' previously using \code{\link[DeProViR]{ModelTraining}}
   #' @importFrom keras load_model_weights_hdf5
   #' @importFrom data.table fread
   #' @export
   #' @examples
   #' Loading_trainedModel <- Load_PreTrainedModel()


   Load_PreTrainedModel <-
      function(input_dim = 20,
               output_dim = 100,
               filters_layer1CNN = 32,
               kernel_size_layer1CNN = 16,
               filters_layer2CNN = 64,
               kernel_size_layer2CNN = 7,
               pool_size = 30,
               layer_lstm = 64,
               units = 8,
               metrics = "AUC",
               filepath = system.file("extdata", "Pre_trainedModel",
                                      package = "DeProViR")){


         #### network architecture

         ## viral side
         embedding_matrix_v <-
            fread(system.file("extdata","Pre_trainedModel","viral_embedding.csv",
                                 package = "DeProViR"))
         embedding_matrix_v <- as.matrix(embedding_matrix_v)

         seq1 <-
            layer_input(shape = 1000, name = "viral_seq")


         m1 <-
            seq1 %>%
            layer_embedding(input_dim = input_dim,
                            output_dim = output_dim,
                            weights = list(embedding_matrix_v),
                            trainable = FALSE,
                            input_length = 1000) %>%
            layer_conv_1d(filters = filters_layer1CNN,
                          kernel_size =kernel_size_layer1CNN,
                          strides = 2,
                          activation = "relu") %>%
            layer_conv_1d(filters = filters_layer2CNN,
                          kernel_size =kernel_size_layer2CNN,
                          strides = 2,
                          activation = "relu") %>%
            layer_max_pooling_1d(pool_size = pool_size) %>%
            bidirectional(layer_lstm(units = layer_lstm)) %>%
            layer_dropout(rate = 0.3)


         ## host side
         embedding_matrix_h <-
            fread(system.file("extdata","Pre_trainedModel","host_embedding.csv",
                                 package = "DeProViR"))
         embedding_matrix_h <- as.matrix(embedding_matrix_h)


         seq2 <-
            layer_input(shape = 1000, name = "host_seq")

         m2 <-
            seq2 %>%
            layer_embedding(input_dim = input_dim,
                            output_dim = output_dim,
                            weights = list(embedding_matrix_h),
                            trainable = FALSE,
                            input_length = 1000) %>%
            layer_conv_1d(filters = filters_layer1CNN,
                          kernel_size =kernel_size_layer1CNN,
                          strides = 2,
                          activation = "relu") %>%
            layer_conv_1d(filters = filters_layer2CNN,
                          kernel_size =kernel_size_layer2CNN,
                          strides = 2,
                          activation = "relu") %>%
            layer_max_pooling_1d(pool_size = pool_size) %>%
            bidirectional(layer_lstm(units = layer_lstm)) %>%
            layer_dropout(rate = 0.3)

         merge_vector <-
            layer_concatenate(list(m1,m2))
         out <-
            layer_dense(units = units, activation = "relu")(merge_vector) %>%
            layer_dense(units = 1, activation = "sigmoid")
         merge_model <-
            keras_model(inputs=list(seq1,seq2),outputs=out)
         merge_model %>% compile(
            optimizer = "adam",
            loss = "binary_crossentropy",
            metrics = metrics
         )

         filepath <-
            file.path(filepath, "pre_trained_glove_model_PubTrained_final_cv.h5")

         merge_model %>%
            load_model_weights_hdf5(merge_model,
                                    filepath = filepath)


         return(merge_model)
      }



   #'predInteractions
   #'@title Predict Unknown Interactions
   #' @param url_path  URL path to GloVe embedding. Defaults to
   #' "https://nlp.stanford.edu/data/glove.6B.zip".
   #'@param Testingset A data.frame containing unknown interactions. For demo,
   #'we can use the file in extdata/test_Set.
   #'@param trainedModel Pre-trained model stored in extdata/Pre_trainedModel
   #'or the training model "$merge_model" achieved by
   #'\code{\link[DeProViR]{ModelTraining}}.
   #'@description This function initially constructs an embedding matrix from
   #'the viral or host protein sequences and then predicts scores for unknown
   #'interactions. Interactions with scores greater than 0.5 are more likely
   #'to indicate interaction.
   #'@return Probability scores for unknown interactions
   #'@export
   #'@examples
   #' trainedModel <- Load_PreTrainedModel()
   #' # load test set (i.e., unknown interactions)
   #' testing_set <- data.table::fread(
   #' system.file("extdata", "test_Set", "test_set_unknownInteraction.csv",
   #' package = "DeProViR"))
   #' # now predict interactions
   #' options(timeout=240)
   #' predInteractions <-
   #'  predInteractions(url_path = "https://nlp.stanford.edu/data",
   #'  testing_set, trainedModel)

   predInteractions <-
      function(url_path = "https://nlp.stanford.edu/data",
               Testingset,
               trainedModel) {

         #### Glove importing
         embeddings_index <-
            gloveEmb_import(url_path)

         message("GLoVe importing is done ....")

         #### viral embedding
         viral_embedding <-
            encode_ViralSeq(Testingset, embeddings_index)

         message("Viral Embedding is done ....")


         #### host embedding
         host_embedding <-
            encode_HostSeq(Testingset, embeddings_index)

         message("Host Embedding is done ....")

         #prediction on test set
         x_pred <- as.matrix(cbind(viral_embedding$data_v,
                                   host_embedding$data_h))

         prob <- trainedModel %>% predict(list(x_pred[,1:1000],
                                               x_pred[,1001:2000]))

         return(prob)



      }





