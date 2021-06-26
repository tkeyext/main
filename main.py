from model.train import LogisticRegression


regressor = LogisticRegression()
regressor.load("model.txt")

txt = """Sinema salonları 1 Temmuz’a kadar kapalı kalmaya devam edecek. Yayımlanan
genelgede Kültür ve Turizm Bakanlığı yetkilileri ve sektör temsilcileriyle
yapılan görüşmeler sonucunda salgınla mücadele tedbirleri kapsamında
devamlılığın sağlanması hususu göz önünde bulundurularak bu kararın alındığı
belirtildi. 

 * Bir adım geriden: 1 Haziran’da yayımlanan genelgede sinema salonlarının pazar
   günleri hariç her gün 07.00-21.00 saatleri arasında %50 kapasiteyle faaliyet
   gösterebileceği belirtilmiş bağımsız sinema salonları hafta sonu özel
   gösterimler düzenlemek için çalışmalarına başlamıştı."""

regressor.predict
