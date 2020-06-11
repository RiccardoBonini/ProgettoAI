# ProgettoAI
Elaborato per l'esame di Intelligenza Artificiale

Il file Main.py contiene l'esecuzione dell'esperimento:
- Vengono importati tutti gli altri file, assieme alle librerie sklearn.metrics (per le funzioni accuracy_score e classification_report), numpy per poter utilizzare i numpy_array, e matplotlib per disegnare il grafico degli errori per ogni epoca con plot.
- Definiamo la funzione permute per poter "mescolare" il training set.
- Carichiamo il dataset tramite mnist_reader.load_mnist.
- Facciamo partire il ciclo sui due casi. Ad ogni iterazione, selezioniamo le due classi e preleviamo gli elementi di tali classi dal dataset originario, generando un dataset binario. Mescoliamo il training set, e alleniamo il Voted Perceptron su 3 dimensioni diverse tramite la funzione FIT, ogni volta chiamando la funzione VOTE sul test set, stampando accuracy e classification report.
- Completati i 3 allenamenti + votazioni sulle 3 dimensioni, stampiamo il grafico Test Errors tramite la funzione plot.

Il tempo complessivo di esecuzione è di circa 20 minuti su un processore Intel Core i5-7400 e 8 GB di RAM.
