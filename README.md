## DeProViR
Emerging infectious diseases, exemplified by the zoonotic COVID-19 pandemic 
caused by SARS-CoV-2, are grave global threats. Understanding protein-protein 
interactions (PPIs) between host and viral proteins is essential for therapeutic 
targets and insights into pathogen replication and immune evasion. While 
experimental methods like yeast two-hybrid screening and mass spectrometry 
provide valuable insights, they are hindered by experimental noise and costs, 
yielding incomplete interaction maps. Computational models, notably DeProViR, 
predict PPIs from amino acid sequences, incorporating semantic information 
with GloVe embeddings. DeProViR employs a Siamese neural network, integrating 
convolutional and Bi-LSTM networks to enhance accuracy. It overcomes 
limitations of feature engineering, offering an efficient means to predict 
host-virus interactions, which holds promise for antiviral therapies and 
advancing our understanding of infectious diseases.



## Installation
To use this package, the initial step involves installing both TensorFlow 
and Keras in Python, followed by establishing a connection to R. 
You can refer to the official TensorFlow documentation 
(https://tensorflow.rstudio.com) and the Keras documentation 
(https://keras.rstudio.com) for detailed instructions on these 
installations and connecting R with these libraries.

You can then install the `DeProViR` from bioconductor using:

```r
if(!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager") 
}
BiocManager::install("DeProViR")
```

To view documentation for the version of this package installed in your 
system, start R and enter:

```r
browseVignettes("DeProViR")
```

To install the development version in `R`, run:
  
```r
if(!requireNamespace("devtools", quietly = TRUE)) {
  install.packages("devtools") 
}
devtools::install_github("mrbakhsh/DeProViR")
```


## Contribute

Check the github page for [source code](https://github.com/mrbakhsh/DeProViR)

## License
This project is licensed under the MIT License - see the LICENSE.md 
file for more details.


