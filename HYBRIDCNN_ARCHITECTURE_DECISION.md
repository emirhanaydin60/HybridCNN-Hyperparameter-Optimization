# HybridCNN Giriş Çözünürlüğü ve Mimari Karar Raporu

## Kapsam ve yöntem

Bu rapor mevcut kaynak kodu, optimizasyon özetleri ve deney loglarının
incelemesine dayanır. Kod değiştirilmemiş, eğitim başlatılmamış ve deney
dosyaları değiştirilmemiştir. Seçenek B'deki boyut tablosu yalnızca olası bir
tasarım örneğidir; mevcut koda uygulanmış değildir.

Başlıca kaynaklar:

- [`model.py`: Conv/Residual/Inception-Lite tanımları](./model.py#L9)
- [`model.py`: SE, son pool ve forward sırası](./model.py#L87)
- [`data_loader.py`: CIFAR-10 boyutu](./data_loader.py#L40)
- [`data_loader.py`: ISIC-2019 boyutu ve dönüşümler](./data_loader.py#L63)
- [`data_loader.py`: Brain Tumor boyutu ve dönüşümler](./data_loader.py#L82)
- [`scripts/run_final_experiments.py`: final ImageFolder dönüşümleri](./scripts/run_final_experiments.py#L204)
- [`metrics.py`: eğitim forward/backward yolu](./metrics.py#L12)
- [`config.py`](./config.py)
- [`GWO Run1 optimized hyperparameters`](./results/main_experiment/GWO/run_01/summary.json#L119)
- [`GWO Run3 optimized hyperparameters`](./results/main_experiment/GWO/run_03/summary.json#L119)
- [`WOA Run3 optimized hyperparameters`](./results/main_experiment/WOA/run_03/summary.json#L119)
- [`CIFAR-10 GWO_Run1 batch observations`](./results/final_experiments/CIFAR10/HybridCNN_GWO_Run1/run.log#L1)
- [`ISIC-2019 GWO_Run1 OOM record`](./results/final_experiments/ISIC2019/HybridCNN_GWO_Run1/run.log#L1)
- [`Brain Tumor GWO_Run1 OOM record`](./results/final_experiments/BrainTumor/HybridCNN_GWO_Run1/run.log#L1)

## 1. Mevcut mimarinin doğrulanmış özeti

### Veri giriş boyutları

| Veri kümesi | Kanallar | Model girişinin H×W boyutu | Kaynaktaki yol |
|---|---:|---:|---|
| CIFAR-10 | 3 | 32×32 | `_build_dataset` `img_size=32` döndürür; CIFAR transform'u `ToTensor` ve normalize işlemleridir. |
| ISIC-2019 | 3 | 224×224 | ImageFolder transform'u `Resize(256)`, `CenterCrop(224)`, `ToTensor`; `img_size=224`. |
| Brain Tumor | 3 | 224×224 | ImageFolder transform'u `Resize(256)`, `CenterCrop(224)`, `ToTensor`; `img_size=224`. |

Dolayısıyla beklenen tensör şekilleri sırasıyla `[B, 3, 32, 32]` ve
`[B, 3, 224, 224]`'tür. Burada `B`, batch size değeridir.

Brain Tumor veri yükleyicisinin eski sürümü `TEST` klasörünü validation olarak
da kullanıyordu. Bu, giriş boyutunu değiştirmez; ancak eski Brain Tumor
validation/test sonuçlarını bağımsız test performansı olarak yorumlamayı
geçersiz kılar.

### Convolution konfigürasyonu ve uzamsal boyut

`model.py` içindeki `nn.Conv2d` çağrılarında `stride` belirtilmemiştir; PyTorch
varsayılanı `stride=1`'dir. `padding` ve `dilation` şu şekildedir:

| Bölüm | Kernel | Stride | Padding | Dilation | Uzamsal etkisi |
|---|---:|---:|---:|---:|---|
| Stem | Optimize edilen `shared_conv_kernel_size` (3 veya 5) | 1 (varsayılan) | `_same_padding(k,d) = ((k-1)//2)×d` | Optimize edilen `dilation`; seçili üç run'da 1 | Giriş H×W korunur |
| Residual block, conv1 ve conv2 | Aynı optimize edilen `shared_conv_kernel_size` | 1 (varsayılan) | Aynı `same` formülü | Optimize edilen `dilation`; seçili run'larda 1 | Blok H×W'yi korur; residual toplama şekilleri eşleşir |
| Inception-Lite branch 1 | 1×1 | 1 (varsayılan) | 0 (belirtilmemiş, varsayılan) | 1 | H×W korunur |
| Inception-Lite branch 2 | 3×3 | 1 (varsayılan) | `_same_padding(3,d)=d` | `dilation` argümanı | H×W korunur |
| Inception-Lite branch 3 | 5×5 | 1 (varsayılan) | `_same_padding(5,d)=2d` | `dilation` argümanı | H×W korunur |
| SE channel gate | `AdaptiveAvgPool2d(1)` | — | — | — | Gate tensörü 1×1 olur; sigmoid ağırlıkları H×W feature map ile çarpıldığından ana akışın boyutu küçülmez |
| Son `self.pool` | `AdaptiveAvgPool2d(1)` | — | — | — | İlk gerçek ana-akış uzamsal indirgemesi: H×W → 1×1 |

Mimari sırası stem → residual1 → Inception-Lite → residual2 → SE → son
adaptive average pool şeklindedir. Stem, residual bloklar ve Inception-Lite
içinde stride veya pooling ile downsampling yoktur. Bu nedenle her veri
kümesinde son havuzlamadan hemen önceki feature map boyutları:

| Veri kümesi | Stem sonrası | Residual1 sonrası | Inception-Lite sonrası | Residual2/SE sonrası | Son pool sonrası |
|---|---:|---:|---:|---:|---:|
| CIFAR-10 | 256×32×32 | 256×32×32 | 256×32×32 | 256×32×32 | 256×1×1 |
| ISIC-2019 | 256×224×224 | 256×224×224 | 256×224×224 | 256×224×224 | 256×1×1 |
| Brain Tumor | 256×224×224 | 256×224×224 | 256×224×224 | 256×224×224 | 256×1×1 |

Tablodaki kanal sayısı seçili optimize edilmiş modellerde
`base_filters=256` olduğu içindir. Daha genel modelde bu değer değişebilir.
`AdaptiveAvgPool2d(1)` öncesinde yüksek çözünürlüklü iki veri kümesinde 224×224
boyutlu feature map'ler tutulmaktadır; yalnızca classifier'a giden son tensör
1×1'e iner.

### Üç seçili HybridCNN konfigürasyonu

Eski CIFAR-10 optimizasyon özetlerinde:

| Model | Kernel | Base filters | Dilation | Final neurons | SE ratio | Optimize edilmiş batch size | Learning rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| HybridCNN_GWO_Run1 | 5 | 256 | 1 | 256 | 8 | 32 | 0.00123516 |
| HybridCNN_GWO_Run3 | 3 | 256 | 1 | 256 | 4 | 32 | 0.00145987 |
| HybridCNN_WOA_Run3 | 5 | 256 | 1 | 128 | 4 | 32 | 0.00141881 |

Bu sayılar `best_hyperparameters` kayıtlarından alınmıştır. Üçünde de
`base_filters=256`, `dilation=1` ve batch size 32 görünür. Proje genelindeki
`config.py` varsayılan batch size'ı 128'dir; bu değer seçili optimizasyon
sonuçlarındaki gerçek/tuned batch size ile karıştırılmamalıdır.

Ek log ayrımı: daha sonraki CIFAR-10 tanılama loglarında GWO_Run1 için batch 8
ve batch 128 denemeleri; GWO_Run3 ve WOA_Run3 için batch 8 model-kurulum
kayıtları vardır. Bunlar optimizer summary'deki tamamlanmış CIFAR-10 run'larının
batch ayarını değiştirmez ve her biri tamamlanmış nihai eğitim olarak
değerlendirilmemelidir. Karşılaştırma için en doğrudan optimize edilmiş batch
kanıtı, yukarıdaki summary kayıtlarındaki 32'dir.

### `base_filters` ve aktivasyon belleği

Bir feature map'in eleman sayısı `B × C × H × W` ile, saklama boyutu da veri
tipindeki byte/eleman ile orantılıdır. `base_filters=256`, stem, residual
blokları ve Inception-Lite çıkışındaki ara kanal sayısını 256 yapar. Kanal
sayısının artması bu ara tensörlerin ve bunlarla ilişkili geri yayılım
aktivasyonlarının belleğini yaklaşık doğrusal artırır; uzamsal H ve W'nin
artması ise maliyeti H×W alanıyla artırır.

Ölçek örneği — batch 32, 256 kanal, FP32 (eleman başına 4 byte) varsayımıyla,
tek bir tam feature map için:

| Feature map | Tek tensör büyüklüğü |
|---|---:|
| `[32,256,32,32]` | 32 MiB |
| `[32,256,224,224]` | 1.53125 GiB |

224/32 = 7 kat kenar uzunluğu; alan farkı 7² = **49 kat**tır. Bu hesap tek
feature map içindir; eğitim sırasında autograd için birden fazla ara aktivasyon,
gradient'ler, optimizer durumları ve convolution workspace'leri de gerekir.
Dolayısıyla bu tablo toplam VRAM tüketimi değildir.

## 2. OOM kayıtları ile ilişki

ISIC-2019 ve Brain Tumor için ilk HybridCNN denemelerinde loglar batch size 32
ile başlandığını, GPU'nun 12 GiB toplam kapasite bildirdiğini, 0 bytes boş
gördüğünü ve backward sırasında ek 1.53 GiB allocation isteğinin başarısız
olduğunu kaydediyor. Hata stack'i `loss.backward()` (`metrics.py`) noktasına
ve ardından C++ autograd motoruna gider; ilk hatalarda belirli bir Python
convolution katmanı kesin olarak isimlendirilmemiştir. Ayrı bir ISIC yüksek
batch retry stack'i `ResidualBlock` içindeki conv2 forward yoluna kadar
izlenmiştir.

İlk hata mesajlarındaki PyTorch allocated değerleri 23.27–25.27 GiB
aralığındadır; aynı mesajlar aygıt kapasitesini 12.00 GiB olarak verdiği için
bu iki değer birbiriyle tutarlı değildir. Logda 0 bytes free ve yaklaşık
1.15–2.73 GiB reserved-but-unallocated da yazılıdır. Tarihsel GPU telemetrisi
ve process bazında snapshot saklanmadığından bu muhasebe tutarsızlığının
nedenini mevcut kayıtlardan saptamak mümkün değildir.

Eski `metrics.py` ve deney eğitim yolunda autocast/GradScaler/AMP çağrısı
bulunmuyor; training'de `logits = model(xb)` ve ardından `loss.backward()`
doğrudan çalışıyor. Dolayısıyla bu OOM kayıtları mixed precision ile alınmış
sonuçlar değildir. Daha sonraki ayrı Brain Tumor denemesinde AMP ve daha küçük
mikro-batch ile ilk epoch tamamlanmış, fakat çok yavaş olduğu için süreç
durdurulmuştur. Bu farklı koşul ilk OOM kayıtlarıyla karıştırılmamalıdır.

## 3. Seçenek A — Özgün mimariyi korumak

Bu seçenekte CIFAR-10'da optimize edilen topoloji ve hiperparametreler
değişmeden kalır; ISIC-2019/Brain Tumor için yalnızca çalışma zamanı ve bellek
yönetimi ele alınır.

### Teknik değerlendirme

- Mimari, kernel/dilation/base_filters/final_neurons/SE ratio ve optimize
  edilen learning rate korunur.
- AMP, küçük mikro-batch ve gradient accumulation OOM riskini azaltabilir.
  Örneğin mikro-batch 8 ve accumulation 4, optimizer adımı başına 32 örneğin
  gradient katkısını biriktirebilir.
- Gradient accumulation, BatchNorm davranışını batch 32 ile aynı yapmaz:
  BatchNorm her mikro-batch için ayrı istatistik hesaplar. Dolayısıyla
  efektif batch sayısı 32 olsa bile mikro-batch 8'in normalizasyon davranışı
  gerçek batch32 ile özdeş değildir.
- AMP sayısal hassasiyeti değiştirir; bu bir mimari değişiklik değildir, ancak
  eğitim sayısal protokolünde kaydedilmesi gereken bir farklılıktır.
- Önceki BT kontrollü denemesinde AMP ile batch 8 ve batch 16 çalışmış; batch
  16/accumulation 2 için yaklaşık 11 dakika/epoch ölçülmüştür. Bu, çözünürlük
  ve modelin değiştirilmeden çalıştırılabildiğine dair sınırlı bir pratik
  kanıttır; bu süre CIFAR ile karşılaştırılabilir benchmark değildir.

### Avantaj ve riskler

**Avantajlar**

- Araştırma sorusundaki "CIFAR-10'da optimize edilmiş aynı modeli hedef veri
  kümelerine aktarma" anlamını korur.
- CIFAR-10 sonucu ile yüksek çözünürlüklü hedef sonuç arasındaki farkın yeni
  bir topoloji değişikliğinden kaynaklanması önlenir.
- Daha küçük mikro-batch/AMP gibi hesaplama önlemleri açıkça raporlandığında
  mimari aktarımı ile hesaplama kolaylığı birbirinden ayrılabilir.

**Dezavantajlar ve riskler**

- Aktiviteler 224×224 boyunca yüksek kanal sayısıyla tutulduğu için GPU
  belleği ve epoch süresi yüksektir.
- Çok küçük mikro-batch BatchNorm istatistiklerini ve dolayısıyla optimizasyon
  davranışını etkileyebilir.
- AMP ve accumulation, CIFAR-10'daki özgün eğitim numeriği/batch uygulamasıyla
  tamamen aynı yürütme koşulu değildir.
- OOM'un kaynağını kesin bir tek convolution'a bağlayan ayrıntılı profiler
  kaydı mevcut değildir; OOM backward aşamasında saptanmıştır.

## 4. Seçenek B — Kontrollü downsampling eklemek

Bu seçenek yüksek çözünürlüklü tensörleri ilk katmanlardan sonra kademeli
küçültür. Aşağıdaki örnek yalnızca tasarım hesabıdır: stem'de stride 2 ve
ardından residual1 ile Inception-Lite arasındaki iki stage sınırında 2×2
pooling varsayılmıştır. Bu mimari mevcut kodda yoktur.

| Aşama | Örnek işlem (uygulanmış değildir) | CIFAR benzeri girişten boyut | 224×224 girişten boyut |
|---|---|---:|---:|
| Giriş | Mevcut tensor | 32×32 | 224×224 |
| Stem çıkışı | Stem conv stride 2 varsayımı | 16×16 | 112×112 |
| Residual1 | Mevcut blok stride 1 | 16×16 | 112×112 |
| Stage indirgeme 1 | Residual1 sonrası 2×2 pool varsayımı | 8×8 | 56×56 |
| Inception-Lite | Mevcut branch'ler stride 1 | 8×8 | 56×56 |
| Stage indirgeme 2 | Inception sonrası 2×2 pool varsayımı | 4×4 | 28×28 |
| Residual2 + SE | Mevcut blok/gate, boyut korunur | 4×4 | 28×28 |
| Son adaptive average pool | Mevcut son pool | 1×1 | 1×1 |

Bu örnekte stride-2 stem çıkışındaki tek 256-kanallı FP32 feature map, batch32
için 224 çözünürlükteki 1.53125 GiB yerine 112 çözünürlükte yaklaşık
0.3828 GiB olur. Sonraki 56 ve 28 boyutlardaki tek feature map'ler sırasıyla
yaklaşık 0.0957 GiB ve 0.0239 GiB olur. Bunlar yalnızca tensör boyutu
hesaplarıdır; toplam VRAM veya hızlanma garantisi değildir. Stem öncesi giriş
ve stem convolution workspace/geri yayılım maliyetleri kalır.

### Beklenen etkiler

- **Bellek/hesaplama:** Sonraki convolution'ların çıktı alanı küçülür; sabit
  kanal ve kernel için uzamsal konum sayısıyla ilişkili aktivasyon ve işlem
  maliyetinin azalması beklenir. Gerçek toplam tasarruf profiling yapılmadan
  kesin nicelenemez.
- **Özellik çıkarımı:** Erken stride/pooling küçük lezyon, kenar, doku veya
  tümör sınırı detaylarını azaltabilir. Daha geniş receptive field'a daha
  düşük çözünürlükte erişim sağlarken ince detay kaybı riski doğar.
- **Kapasite/anlam:** Parametre sayısı aynı kalabilir veya projection/ek
  işlemler eklenirse değişebilir; fakat temsil edilen uzamsal bilgi ve katman
  başına receptive field değişir. Aynı parametre sayısı, aynı mimari veya aynı
  öğrenme problemi demek değildir.
- **Skip bağlantıları:** Downsampling residual bloğun içine stride ile
  konulursa residual ve identity yollarının H×W/kanal şekilleri eşleştirilmek
  zorundadır; projection shortcut veya blok dışında havuzlama gibi açık bir
  tasarım gerekir.
- **Kernel/dilation anlamı:** CIFAR-10'da öğrenilen kernel ve dilation
  değerleri farklı ölçeklerde feature map'lere uygulanacağından etkin
  receptive field ve hiperparametre yorumu değişir.

## 5. Bilimsel geçerlilik ve adil karşılaştırma

### Hiperparametre aktarımı mı, mimari değişikliği mi?

Stride veya pooling eklemek **mimari değişikliğidir**; yalnızca
hiperparametre aktarımı değildir. Bu değişiklik feature map çözünürlüğünü,
receptive field gelişimini, skip-connection tasarımını ve konvolüsyonların
gördüğü sinyal ölçeğini değiştirir.

Downsampling sonrası CIFAR-10'da optimize edilen kernel, dilation ve
`base_filters` değerleri aynı sayısal değerlerde kalsa da artık aynı işlevsel
anlama sahip olmaz. Bu nedenle sonuç "özgün CIFAR-10 optimize HybridCNN'in
doğrudan hedef veri kümesine aktarımı" diye sunulmamalı.

Mimari değiştirilirse model ayrı bir sürüm/isim altında, örneğin
`HybridCNN-DS` olarak izlenmelidir. Özgün HybridCNN ve downsample edilmiş
sürümün checkpoint, config ve sonuçları birbirine karıştırılmamalıdır.

### Sabit tutulması gereken koşullar

Modeller arası karşılaştırmada, veri protokolünün izin verdiği ölçüde şu
koşullar önceden belirlenip kaydedilmelidir:

- aynı train/validation/test ayrımları, split seed ve sınıf eşlemesi;
- test kümesinin yalnızca son değerlendirme için kullanılması;
- aynı hedef giriş çözünürlüğü ve dönüşüm/normalizasyon;
- aynı epoch üst sınırı, early-stopping ölçütü ve patience;
- optimizer, learning-rate politikası, weight decay ve varsa augmentation;
- aynı rastgele seed politikası ve model seçimi kuralı;
- pretrained ağırlık kullanımının açık ve modeller arasında karşılaştırılabilir
  olması;
- değerlendirmede aynı sınıf sırası ve aynı metrik tanımları;
- AMP, mikro-batch, gradient accumulation ve effective batch ayarlarının
  kaydı.

Baseline mimarileri doğası gereği farklıdır; adil karşılaştırma, topolojileri
aynı yapmak değil, veri ayrımı/evaluation protokolü ve eğitim bütçesini şeffaf
ve mümkün olduğunca eşdeğer tutmaktır. Gradient accumulation ile effective
batch eşitlenirken BatchNorm mikro-batch farkı ayrıca raporlanmalıdır.

### Seçeneklerin araştırma sorusuna göre anlamı

- Araştırma önceliği **CIFAR-10'da optimize edilen aynı HybridCNN'in transfer
  edilebilirliğini** ölçmekse, Seçenek A bu iddiaya daha doğrudan cevap verir.
  Bedeli yüksek bellek ve eğitim süresi ile olası küçük mikro-batch/AMP
  etkileridir.
- Araştırma önceliği **224×224 görüntülerde pratik ve ölçeklenebilir bir
  HybridCNN tasarlamak** ise, Seçenek B makul bir yeni model ailesi olabilir.
  Bu durumda iddia "aynı optimize edilmiş model" değil, "downsampling ile
  değiştirilmiş model" olur.
- Her iki soruyu yanıtlamak istenirse özgün ve DS sürümü aynı hedef veri
  ayrımları, epoch bütçesi, metrikler ve test protokolüyle ayrı deney kolları
  olarak karşılaştırılabilir. Bu yalnızca olası takip yaklaşımıdır; burada
  deney tasarlanmamış veya çalıştırılmamıştır.

## 6. Karar özeti: avantajlar, dezavantajlar ve riskler

| Başlık | Seçenek A: özgün topoloji | Seçenek B: downsampling |
|---|---|---|
| CIFAR optimizasyonunun aktarım anlamı | Korunur | Değişir; doğrudan mimari aktarım değildir |
| Yüksek çözünürlükte aktivasyon belleği | Yüksek kalır | Downsampling sonrası azalması beklenir |
| Küçük uzamsal detay | Daha çok korunur | Erken küçültmeyle kaybolabilir |
| Eğitim uygulanabilirliği | AMP/microbatch gerekebilir; uzun sürer | Daha düşük maliyet olası, ölçülmeli |
| Bilimsel adlandırma | Aynı HybridCNN konfigürasyonu | Ayrı mimari sürüm olarak raporlanmalı |
| Başlıca risk | OOM/süre ve küçük mikro-batch BN etkisi | Performans ve aktarım iddiası yeni mimariyle karışabilir |

**Gerekçeli öneri:** Öncelikli araştırma sorusu CIFAR-10'da optimize edilen
konfigürasyonların ISIC-2019 ve Brain Tumor'a aktarımı olduğundan, ilk bilimsel
karşılaştırmanın referans kolu Seçenek A olmalıdır. AMP ve accumulation gibi
optimizasyonlar mimariyi korurken denenebilir; ancak AMP ve mikro-batch
farkları raporlanmalı, BatchNorm etkisi göz önünde tutulmalıdır. Bu kol hedef
donanımda sürdürülebilir değilse Seçenek B, kaynak kısıtını örtük biçimde
çözmek yerine açıkça yeni bir `HybridCNN-DS` mimarisi olarak ve farklı bir
araştırma sorusuyla değerlendirilmelidir. OOM riskinin çözümü için mimariyi
değiştirme kararı, bilimsel hedef ve gerekiyorsa yeni optimizasyon/ablation
protokolü belirlenmeden verilmemelidir.

## 7. Karar öncesi eksik bilgiler

1. Başarılı CIFAR run'larında peak allocated/reserved VRAM telemetrisi mevcut
   değil; CIFAR ile hedef veri kümelerinin nicel peak-memory karşılaştırması
   yapılamıyor.
2. Eski OOM loglarındaki `allocated by PyTorch` değerleri 12-GiB aygıt
   kapasitesini aşıyor. Process/GPU telemetry bulunmadığı için bu tutarsızlık
   açıklanamıyor.
3. İlk OOM traceback'leri belirli bir Python convolution alt-katmanını değil
   `loss.backward()`/autograd'ı gösteriyor. Ayrıntılı profiler/katman bazlı
   memory trace yok.
4. Gradient accumulation kullanılan küçük mikro-batch'lerde BatchNorm'un
   nasıl ele alınacağına ilişkin önceden tanımlı bir protokol yok.
5. Hedef veri kümeleri için aynı 50-epoch bütçesinin pratik uygulanabilirliği
   sınırlı; ancak epoch bütçesini değiştirmek ayrı ve önceden kararlaştırılmış
   bilimsel protokol gerektirir.

Bu rapor yalnızca mimari inceleme ve karar desteğidir; hiçbir topoloji
değişikliği veya yeni eğitim içermez.
