# Optical Character Recognition

Digit and letter recognition neural networks using MNIST and EMNIST datasets. Vanilla, convolutional and convolutional-LSTM networks were used for clasifying individual characters and words from artificially created images from the combined EMNIST letter images.

## Prerequisites

 - Libraries: __TensorFlow, matplotlib, tkinter, pickle, opencv, PyQT5, pandas,
 - Data: MNIST and EMNIST binary datasets (found in the data folder)
 - The 
 
 
## Structure of the project



Licenta DUMITRSCU Alexandru 343A3

Arhitecturile folosesc imagini MNIST si EMNIST pentru cifre si litere. Pentru arhitecturile LSTM,
predictiile se vor face folosind date create artificial. Scripturile pentru crearea acestor date
sunt generate_clean.py si word_generator.py. Cel din urma extrage cele mai folosite cuvinte din 
limba engleza si le pune in cuvinte.txt. generate_clean.py genereaza imagini de mai multe caractere
folosindu-se de imaginile din fisierele binare de caractere (emnist-letters-train-images-idx3-ubyte)
generate_clean.py:
	-get_in_order: foloseste dictionarul aranjate.txt si separa literele cu label-uri in majuscule,
	 repsectiv litere mici de 3 niveluri de grosime. 
	-create_smaller_data: creaza date cu o probabilitate crescuta ca in imagini sa se afle litere de
	 grosime medie, si scazuta pentru litere subtiri/groase.
	 Parametrul size al acestei metode indica lungimea imaginilor (nr de caractere/imagine)

Desi functioneaza si cu Tensorflow instalat pe CPU, viteza este mult mai mica. In cazul in care
utilizatorul dispune de o placa video 1050TI sau mai buna, de la NVidia, urmatorii pasi sunt necesari
daca se doreste folosirea cardului grafic:
	http://shitalshah.com/p/installing-tensorflow-gpu-version-on-windows/


1.Pentru functionarea antrenarii retelelor este nevoie de urmatoarele:
	-python 3.5+
	-Tesnorflow
	-Numpy
	-cv2
	-pickle
	-tkinter (in cazul in care se doreste etichetarea in continuare a caracterelor)/rularea script-ului
	 separate_labels.py
Toate aceste pachete pot fi instalate folosind pip.

2.Intre arhitecturile de interes, sunt regasite:
In LSTM/lstm.py:
	-append_small_data: citeste fisierele binare create de generate_clean.py si le adauga in liste
	-one_hot_encoding: pune label-urile in vectori de 27 de caractere [0 0 ... 0] cu 1 pe pozitia literei
	 corespunzatoare
	-small_model_var: creaza implicit un model LSTM pe 3 caractere (pentru un nr. diferit de caractere, 
	 trebuie setat generate_clean sa creeze imagini cu alte lungime si trebuie rulat
multe functii auxiliare: 
	-get_lstm_batch: reutrneaza batch-uri de dimensiunea precizata
	-lstm_shuffle: amesteca datele si e folosit de get_lstm_batch atunci cand se termina o epoca de train
	-read_images/read_labels/convert_to_images/show_images_letters etc. sunt metode auxiliare, folosite
	 in program, comentate si care nu au un rol cruical

sources/arhitecturi_pentru_licenta:
	-aici se regaseste arhitectura folosita in aplicatia grafica.
	-multe metode pentru zgomote, arhitectura simpla, convolutiva, metoda de afisare a imaginilor
	-1chr_digits_model_conv.ckpt este modelul produs automat de conv_architecture
	-conv_architecture: metoda care produce arhitectura folosita si in aplicatie
