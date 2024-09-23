# **Enhancing Public Safety - Implementing YOLOv8 for Real-time Weapon Detection (Celurit)**

- Name : Toni Andreas Susanto
- City : Samarinda, Kalimantan Timur

Table of Contents :

1. [Background]
2. [Introduction to Object Detection]
3. [Working with YOLOv8]
4. [Weapon Detection with YOLOv8 Model]
   1. [Data Collection and Preparation]
   2. [Training Model]
   3. [Evaluation and Performance Analysis of Object Detection Models]
   4. [Perform Object Detection on a Video]


## 1. Background <a id="0" ></a>

Tingkat kejahatan yang semakin meningkat menyebabkan keselamatan masyarakat semakin mengkhawatirkan. Oleh karena itu, dibutuhkan solusi inovatif untuk meningkatkan keamanan dan menjamin ketenangan masyarakat. Project ini dirancang agar dapat menerapkan teknik machine learning tingkat lanjut, **Object Detection**, dan menggunakannya untuk menyelesaikan masalah pada dunia nyata. Dengan mengeksplorasi YOLOv8 (You Only Look Once), sebuah algoritma object detection yang terkenal dengan kecepatan dan akurasinya yang luar biasa, peserta akan mengeksplorasi dan menerapkan sistem yang mampu mendeteksi kejahatan dalam berbagai situasi. Harapannya model yang dihasilkan dapat digunakan pada daerah-daerah yang rawan kejahatan dan kriminalitas sehingga pihak berwenang dapat mengambil langkah tegas dan cepat apabila terjadi tindak kejahatan. Alhasil dampak yang dirasakan masyarakat dapat diminimalisir, pelaku kejahatan dapat ditangkap segera dan juga menurunkan tingkat kejahatan masyarakat karena potensi penjahat tertangkap lebih tinggi. Model ini nantinya akan mengenali berbagai senjata sehingga ketika terdeteksi senjata tajam maka akan digunakan untuk melaporkan pada pihak berwenang. 


## 2. Introduction to Object Detection <a id="1" ></a>

Seiring kemajuan perkembangan teknologi **computer vision**, kemampuan komputer untuk memahami dunia visual telah meningkat. Salah satu aplikasi paling fundamental dan serbaguna dalam computer vision adalah **image classification**, dimana sebuah sistem dapat belajar untuk mengidentifikasi sebuah gambar untuk diklasifikasikan ke dalam satu atau lebih kategori. 

Namun, kini computer vision sudah berkembang lebih jauh lagi. Selain mengenali apa objek yang ada dalam sebuah gambar, tetapi juga dapat menentukan lokasi dan ukuran objek tersebut. Task tersebut dapat dilakukan oleh algoritma **object detection**. Object detection melakukan identifikasi lokasi suatu objek dengan melakukan prediksi **bounding box**, kotak yang membungkus objek terdeteksi dalam sebuah gambar. 

![Object Detection](https://deeplobe.ai/wp-content/uploads/2023/06/Object-detection-Real-world-applications-and-benefits.png)


### 2.1 Object Detection vs Image Classification 

Object detection dan image classification merupakan dua task computer vision yang sering kali dikira serupa, namun sebenarnya memiliki perbedaan yang signifikan. Berikut merupakan perbedaan dari kedua task tersebut:

|              | Image Classification                         | Object Detection                                          |
|--------------|----------------------------------------------|-----------------------------------------------------------|
| Tujuan       | Klasifikasi seluruh bagian gambar            | Identifikasi lokasi dan klasifikasi objek di dalam gambar |
| Output       | Label kategori                               | Label kategori dan koordinat bounding box                 |
| Target       | 1 kategori                                   | Banyak kategori                                           |
| Kompleksitas | Lebih sederhana, waktu komputasi lebih cepat | Lebih kompleks, waktu komputasi lebih lambat              |
| Model        | CNN, ResNet, VGG, Inception                  | YOLO, Faster R-CNN, SSD                                   |



### 2.2 Object Detection with YOLO Algorithm

YOLO (You Only Look Once) adalah algoritma object detection yang populer karena kecepatan dan keakuratannya, sehingga memungkinkan object detection secara real-time. YOLO memiliki metode yang memungkinkan untuk melakukan prediksi objek serta bounding box secara sekaligus. Berbeda dengan algoritma lain yang melakukan prediksi secara berulang untuk mendapatkan lokasi objek yang diinginkan didalam gambar. Berikut merupakan tahap-tahap cara kerja dari algoritma YOLO:
    
![YOLO Algorithm](https://i.ibb.co/xJYTnvT/yolo-algorithm.png)

1. **Input Image to YOLO Model**
    
    Gambar yang ingin diprediksi diberikan pada algoritma YOLO yang terdiri atas arsitektur CNN (Convolutional Neural Network). YOLO akan mempelajari fitur-fitur yang terdapat pada gambar yang diinput.
    
2. **Grid Prediction**

    YOLO membagi gambar menjadi grid cell dengan jumlah S×S. Kemudian, setiap cell grid akan membuat prediksi bounding box. Prediksi yang dilakukan adalah prediksi terhadap objek yang ada pada bagian gambar tiap grid.
    
3. **Remove Predictions with Low Confidence**

    Setiap prediksi yang dihasilkan oleh tiap grid akan memiliki nilai confidence score yang merupakan nilai probability sebuah bounding box berisi jenis objek tertentu. Bounding box dengan confidence score di bawah threshold (default: 0.25) akan dibuang.
    
4. **Non-Max Suppression (NMS)**

    Pada bounding box yang tersisa, terdapat kemungkinan suatu object diprediksi beberapa kali oleh grid yang berbeda, sehingga terjadi tumpang tindih antar bounding box. **NMS** digunakan untuk menghilangkan bounding box yang berlebihan pada satu object yang sama dengan hanya menyimpan kotak dengan confidence score tertinggi (diukur dengan **Intersection over Union** atau **IoU**).
    
5. **Final Output**

    Hasil dari proses tersebut adalah sekumpulan bounding box, di mana masing-masing memiliki label kelas dan confidence score. Bounding box ini adalah prediksi akhir algoritma YOLO untuk objek pada gambar baru.



## 3. Weapon Detection with YOLOv8 Model <a id="2" ></a>

YOLOv8 adalah versi terbaru YOLO dari Ultralytics. Sebagai model termutakhir, YOLOv8 mengembangkan versi-versi sebelumnya, memperkenalkan fitur-fitur baru, serta meningkatkan performa dan efisiensi. Berikut dokumentasi YOLOv8 dari Ultralytics : https://docs.ultralytics.com/.


Aplikasi object detection pada deteksi senjata dalam video atau gambar merupakan studi kasus yang sangat penting dan relevan, khususnya dalam konteks peningkatan keamanan dan pencegahan tindak kejahatan. Kita akan mengimplementasikan YOLOv8 untuk melakukan deteksi senjata. Dalam konteks projek kali ini dapat digunakan untuk mendeteksi senjata **Celurit**.