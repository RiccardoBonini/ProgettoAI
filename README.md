# ProgettoAI
Elaborato per l'esame di Intelligenza Artificiale

Il file Main.py contiene l'esecuzione dell'esperimento:
- Vengono importati tutti gli altri file, assieme alle librerie sklearn.metrics (per le funzioni accuracy_score e classification_report), numpy per poter utilizzare i numpy_array, e matplotlib per disegnare il grafico degli errori per ogni epoca con plot.
- Definiamo la funzione permute per poter "mescolare" il training set.
- Carichiamo il dataset tramite mnist_reader.load_mnist.
- Facciamo partire il ciclo sui due casi (classi 0 e 5, e classi 0 e 7). Ad ogni iterazione, selezioniamo le due classi e preleviamo gli elementi di tali classi dal dataset originario, generando un dataset binario. Mescoliamo il training set, e alleniamo il Voted Perceptron su 3 dimensioni diverse tramite la funzione FIT, ogni volta chiamando la funzione VOTE sul test set, stampando accuracy e classification report.
- Completati i 3 allenamenti + votazioni sulle 3 dimensioni, stampiamo il grafico Test Errors tramite la funzione plot.

Il tempo complessivo di esecuzione è di circa 15 minuti su un processore Intel Core i5-7400 e 8 GB di RAM.
Per la realizzazione di questo progetto, sono state consultate le seguenti fonti:
- Freund Y., Schapire R. E., Large Margin Classification Using the Perceptron Algorithm, 1999
- Manabu Sassano, An Experimental Comparison of the Voted Perceptron and Support Vector Machines in Japanese Analysis Tasks, 2001, per spunti ulteriori sul Kernel Voted Perceptron
- La libreria sklearn per utilizzare le funzioni accuracy_score e classification_report
