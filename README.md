# ProgettoAI
Elaborato per l'esame di Intelligenza Artificiale

Si deve scaricare il file data/fashion dal link https://github.com/zalandoresearch/fashion-mnist ed inserirlo nella directory del progetto, ed i 3 file .py. L'esecuzione del progetto è all'interno del file Main.py. Di seguito come esso si svolge:
- Vengono importati tutti gli altri file, assieme alle librerie sklearn.metrics (per le funzioni accuracy_score e classification_report), numpy per poter utilizzare i numpy_array, e matplotlib per disegnare il grafico del Test Error e Learning Curve con plot.
- Carichiamo il dataset tramite mnist_reader.load_mnist.
- Vengono scelte le classi 1 ed 8 da estrarre dal dataset. Alleniamo il Voted Perceptron su 3 dimensioni diverse ogni volta tramite la funzione FIT, generando il vettore PARTIAL contenente il numero di errori k per ogni decimo di epoca.  Successivamente chiamiamo la funzione VOTE sul test set, generando le previsioni per ogni campione di epoca; calcoliamo gli errori e generiamo il grafico Test Error per la dimensione corrente. Successivamente calcoliamo accuracy e classification report della previsione finale del Voted Perceptron per la dimensione corrente.
- Completati i 3 allenamenti + votazioni sulle 3 dimensioni, stampiamo il grafico Learning Curve tramite la funzione plot.

Il tempo complessivo di esecuzione è di circa 10 minuti su un processore Intel Core i5-7400 e 8 GB di RAM.
Per la realizzazione di questo progetto, sono state consultate le seguenti fonti:
- Il dataset zalandoresearch, https://github.com/zalandoresearch/fashion-mnist
- Freund Y., Schapire R. E., Large Margin Classification Using the Perceptron Algorithm, 1999
- Manabu Sassano, An Experimental Comparison of the Voted Perceptron and Support Vector Machines in Japanese Analysis Tasks, 2001, per spunti ulteriori sul Kernel Voted Perceptron
- La libreria sklearn per utilizzare le funzioni accuracy_score e classification_report
