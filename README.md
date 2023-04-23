# Final Submission : Machine Learning Pipeline - Diabetes Classification
Nama: Maulana Muhammad

Username dicoding: maoelana

![penderita-diabetes-1-768x512](https://user-images.githubusercontent.com/58927608/233813527-c58cb22f-7f68-4076-a0ab-bd2b1f4b0f88.jpg)

[Sumber Gambar](https://blog.peoplespheres.com/en-us/what-problems-do-human-resources-managers-face-every-day)

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [Diabetes prediction dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) |
| Masalah | Diabetes atau penyakit gula adalah penyakit kronis (jangka panjang) yang perlu diwaspadai. Adapun tanda utama dari diabetes adalah meningkatnya kadar gula darah (glukosa) melebihi nilai normal. Diabetes terjadi ketika tubuh pengidapnya tidak lagi mampu mengambil gula (glukosa) ke dalam sel dan menggunakannya sebagai energi. Kondisi ini pada akhirnya menghasilkan penumpukan gula ekstra dalam aliran darah tubuh. Angka kematian yang diakibatkan oleh diabetes terbilang tinggi. Pada tahun 2021 diabetes telah menyebabkan 6,7 juta kematian di dunia pada 2021. Ini berarti ada 1 kematian setiap 5 detik. |
| Solusi machine learning | Untuk mendeteksi diabetes biasanya pasien menjalani pengecekan kadar gula yang ada pada tubuh. Dokter tidak lansung dapat menyimpulkan apakah pasien terkena diabetes hanya dari sekedar pengecekan kadar gula darah. Dengan sistem diabetes classification menggunakan machine learning, dockter diharapkan dapat mendapatkan pengetahuan untuk diabetes pada pasien yang akan digunakan untuk pertimbangan dan pemeriksaan pasien kedepannnya. |
| Metode pengolahan | Pada kasus ini, terdapat sembilan feature dimana delapan akan digunakan untuk features klasifikasi, dan satu sebagai class, terdapat dua categorical feature dan enam numerical, kemudian akan dilakukan split data menjadi 80:20 untuk data train dan eval. Proses transform akan dilakukan renaming untuk feature yang telah ditransform, one hot encoding untuk class data |
| Arsitektur model | Untuk arsitektur model sendiri, untuk tiap layer menggunakan Dense 256, Dense 64, Dense 16 dengan activation relu, kemudian layer terakhir menggunakan Dense 1 dengan activation sigmoid karena akan mengklasifikasi class yang hanya memiliki dua value yaitu diabetes dan tidak diabetes. Untuk model compile menggunakan optimizers Adam dengan learning_rate 0.001, loss binary_crossentropy dengan metrics BinaryAccuracy |
| Metrik evaluasi | Metrik evaluasi yang digunakan yaitu AUC, Precision, Recall, ExampleCount dan BinaryAccuracy |
| Performa model | Untuk performa model, ditinjau dari accuracy dan lossnya, accuracy yang didapatkan terbilang tinggi dengan accuracy 0.97 pada proses training dan validation, dan loss yang didapatkan pada proses training dan validation yaitu 0.0817 sehingga model ini terbilang baik untuk klasifikasi |
| Opsi deployment | Untuk deployment, sistem ini akan dideploy menggunakan platform railway |
| Web app | [diabetes-classification](https://diabetes-classification-production.up.railway.app/v1/models/diabetes-classification-model/metadata)|
| Monitoring | Monitoring pada sistem ini dilakukan menggunakan prometheus dan grafana. Disini hanya dilakukan proses monitoring untuk menampilkan request yang masuk pada sistem yang akan menamplkan status pada tiap request yang dilakukan, pada sistem ini terdapat tiga status yang ditampilkan yaitu apabila proses request pada sistem klasifikasi not found, invalid argument dan proses klasifikasi berhasil ditandakan dengan ok |
