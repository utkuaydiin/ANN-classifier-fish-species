# Balık Türleri Sınıflandırma Yapay Sinir Ağı (ANN) Modeli

Kaggle = https://www.kaggle.com/code/utkuaydiin/fish-species-ann-classifier


Bu projede, balık türlerini sınıflandırmak için bir Yapay Sinir Ağı (ANN) modeli geliştirilmiştir. Model, TensorFlow ve Keras kullanılarak eğitilmiş ve bir dizi balık görüntüsünden tür sınıflandırması yapılmıştır.

## Proje Özeti

Bu projenin amacı, balık türlerini görüntüye dayalı olarak sınıflandırabilen bir makine öğrenmesi modeli oluşturmaktır. ANN tabanlı bu model, aşırı öğrenmeyi (overfitting) engellemek için düzenlileştirme (regularization) ve dropout katmanları kullanılarak optimize edilmiştir. Eğitim sürecinde erken durdurma (early stopping) ve öğrenme oranı azaltma (learning rate scheduling) yöntemleri uygulanmıştır.

## Veri Seti

Veriseti = https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset
Proje kapsamında kullanılan veri seti, farklı balık türlerine ait görüntülerden oluşmaktadır:

- Görüntüler .png formatında saklanmıştır.
- Her bir tür için “ground truth” (GT) etiketleri mevcuttur.
- Eğitim performansını artırmak amacıyla veri seti görüntü artırma teknikleriyle zenginleştirilmiştir.
- **Sınıflar:** 9 farklı balık türü.
- **Görüntü sayısı:** Tüm sınıflarda yaklaşık 9,000 görüntü.

## Model Mimarisi

Bu projede kullanılan model bir Yapay Sinir Ağı (ANN) modelidir ve aşağıdaki katmanlardan oluşmaktadır:

1. **Giriş Katmanı:** 28x28 boyutundaki RGB görüntüler düzleştirilerek giriş vektörü haline getirilmiştir.
2. **Gizli Katmanlar:**
   - 1024, 512, 256, 128 ve 64 nöron içeren Tam Bağlı (Dense) katmanlar.
   - Eğitim sırasında kararlılığı sağlamak için Batch Normalization uygulanmıştır.
   - Aşırı öğrenmeyi önlemek amacıyla %20 Dropout katmanları eklenmiştir.
   - L2 düzenlileştirme (L2 regularization) ile modelin aşırı karmaşık hale gelmesi önlenmiştir.
3. **Çıkış Katmanı:** Softmax aktivasyon fonksiyonu kullanarak 9 sınıfa ait olasılıkları tahmin eden çıkış katmanı.

## Veri Ön İşleme

- **Görüntü boyutlandırma:** Tüm görüntüler 28x28 piksel boyutuna indirgenmiştir.
- **Veri artırma:** Görüntülerin ölçeklenmesi ve doğrulama verisi ayrılması gibi veri artırma teknikleri uygulanmıştır.
- **Veri setinin bölünmesi:** Veri seti %80 eğitim, %20 test olacak şekilde `train_test_split` kullanılarak bölünmüştür.
- `ImageDataGenerator` kullanılarak veri artırma işlemi yapılmış ve veriler model için hazırlanmıştır.

## Eğitim Süreci

- **Optimizasyon:** Model, Adagrad optimizasyon algoritması ile 0.01 başlangıç öğrenme oranı kullanılarak eğitilmiştir.
- **Kayıp fonksiyonu:** Çok sınıflı sınıflandırma problemi olduğu için `categorical_crossentropy` kayıp fonksiyonu kullanılmıştır.
- **Callback'ler:**
  - Erken durdurma (early stopping), doğrulama kaybı iyileşmediğinde eğitimi durdurmak için kullanılmıştır.
  - Öğrenme oranı zamanla azaltılarak modelin daha yavaş ama daha kararlı öğrenmesi sağlanmıştır.

## Değerlendirme

Modeli değerlendirmek için aşağıdaki yöntemler kullanılmıştır:

- **Doğruluk (Accuracy):** Eğitim ve doğrulama setleri üzerinde doğruluk hesaplandı.
- **Karmaşıklık Matrisi (Confusion Matrix):** Modelin her sınıfta nasıl performans gösterdiği, hangi türleri karıştırdığına dair detaylı bilgi sağladı.
- **Sınıf bazlı ölçümler (Per-Class Accuracy):** Her balık türü için doğruluk hesaplandı.
- **Klasifikasyon Raporu (Classification Report):** Precision, recall, ve F1-score gibi metrikler kullanılarak sınıf performansları raporlandı.

### "Loss Over Epochs" (Kayıp Değerleri)

- Eğitim kaybı (Train Loss) ve doğrulama kaybı (Validation Loss) zamanla azalıyor, bu da modelin eğitildikçe daha iyi performans gösterdiğini gösteriyor.
- Başlangıçta kayıplar oldukça yüksek, ancak birkaç epoch sonra hızla düşmeye başlıyor.
- Epoch civarında doğrulama kaybında küçük bir artış görüyorum; bu muhtemelen overfitting sinyali olabilir, ancak bu geçici bir durum gibi görünüyor çünkü daha sonraki epochlarda kayıp tekrar düşüyor.
- Sonlara doğru hem eğitim hem de doğrulama kayıplarının yakınlaştığını görüyorum, bu da modelin hem eğitim setinde hem de doğrulama setinde iyi performans gösterdiğini, ciddi bir overfitting problemi olmadığını gösteriyor.

### "Accuracy Over Epochs" (Doğruluk Oranları)

- Eğitim ve doğrulama doğrulukları başlangıçta hızlı bir artış gösteriyor. Yaklaşık 10-15 epoch sonra doğrulama doğruluğu stabilize oluyor ve çok fazla dalgalanma göstermiyor.
- Eğitim doğruluğu, doğrulama doğruluğuna göre daha hızlı artıyor; bu normal bir durum çünkü model eğitim verisi üzerinde daha iyi öğreniyor.
- Yaklaşık 90-100. epoch sonrasında doğrulama doğruluğu %80 civarında sabitleniyor, eğitim doğruluğu ise biraz daha yüksek, bu da modelin eğitim verisine iyi adapte olduğunu ama doğrulama verisinde de başarılı olduğunu gösteriyor.
- Sonuç olarak, modelim genel olarak iyi bir performans gösteriyor, ancak doğrulama setindeki küçük dalgalanmalar overfitting riski olabileceğini düşündürüyor. Bunu kontrol altında tutmak için early stopping gibi yaklaşımları değerlendirebilirim.

## Sonuçlar

- **Makro Ortalama (Macro Avg):** %98 doğruluk, her sınıfın performansına eşit önem verilerek hesaplanmış ve genel olarak çok iyi bir sonuç. Model, neredeyse tüm sınıflarda istikrarlı bir şekilde iyi performans gösteriyor.
- **Ağırlıklı Ortalama (Weighted Avg):** %97 doğruluk, veri setindeki her sınıfın örnek sayısına göre ağırlıklandırılarak hesaplanmış. Veri dengesizliği olsa bile modelin genelde yüksek performans gösterdiğini söyleyebiliriz.
- **Mükemmel Performans:** Özellikle Black Sea Sprat ve Horse Mackerel sınıflarında model mükemmel sonuçlar vermiş (%100 doğruluk). Bu sınıflarda modelin hiç hata yapmadığı görülüyor.
- **Düşük Performanslı Sınıflar:** Gilt-Head Bream ve Trout sınıflarında doğruluk biraz daha düşük (%93-94). Bu durum, modelin bu türleri diğer sınıflarla karıştırmaya daha yatkın olduğunu gösteriyor.

### Eğitim Seti Sonuçları:

- **F1 Score:** 0.9997
  - F1 skoru, modelin hassasiyet (precision) ve geri çağırma (recall) dengesini gösterir. Eğitimde F1 skoru neredeyse mükemmel, %99.97 ile modelin çok nadiren hata yaptığını söyleyebiliriz. Yani hem yanlış pozitifler (false positives) hem de yanlış negatifler (false negatives) minimum seviyede.
  
- **Accuracy (Doğruluk):** 0.9997
  - Eğitim setindeki doğruluk da %99.97. Bu, modelin sınıflandırma görevini son derece başarılı bir şekilde yerine getirdiğini ve eğitim verilerinin büyük çoğunluğunu doğru sınıflandırdığını gösterir. Modelin eğitimdeki performansı ideal seviyede.
  
- **Loss (Kayıp):** 1.2991
  - Kayıp değeri (loss), modelin ne kadar hatalı tahmin yaptığını ölçer. 1.2991 değeri yüksek gibi görünse de doğruluk ve diğer metriklerin bu kadar yüksek olması, modelin belki de bazı sınıflarda küçük hatalar yapmış olabileceğini ancak bu hataların genel performansa ciddi bir etki etmediğini gösteriyor.
  
- **Precision (Hassasiyet):** 0.9997
  - Hassasiyet, modelin doğru pozitif tahmin yapma yeteneğini ifade eder. %99.97’lik precision, modelin yanlış pozitif tahminlerde bulunma ihtimalinin çok düşük olduğunu gösterir. Eğitimde model, neredeyse her doğru tahmini isabetli bir şekilde yapmış.
  
- **Recall (Geri Çağırma):** 0.9997
  - Recall, modelin doğru şekilde tanıdığı gerçek pozitiflerin oranını gösterir. %99.97'lik bir recall ile model, eğitim setindeki verilerin neredeyse tamamını doğru tanımlamış ve çok az say
