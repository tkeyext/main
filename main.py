from model import train

train.start('./data/training.csv')

from model.regression import LogisticRegression
from bin.extractor import extractKeywords

txt = """• Porsche
[https://www.bloomberg.com/news/articles/2021-06-20/porsche-to-make-high-performance-battery-cells-in-new-venture?sref=4TAH79Na]
elektrikli araçlarında kullanmak üzere pil hücreleri geliştirmek için Almanya
merkezli şarj modülleri üreticisi Custom Cells ile iş birliği yapacaklarını
açıkladı. Küçük ölçekli üretime 2024’te başlanacağı ve pil teknolojilerinin
motor sporlarında test edileceği belirtildi.

 * Öte yandan: 2030'da firmanın küresel satışlarının %80'inden fazlasının
   tamamen veya kısmen elektrikli modeller olacağı tahmin ediliyor."""

regressor = LogisticRegression()
# Select a models from [model1, model2]
regressor.load("model2")

keywords = extractKeywords(txt, regressor)
print(keywords)