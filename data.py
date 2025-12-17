import datetime
import random
import psycopg2
from dotenv import load_dotenv
import os 

load_dotenv()

db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")
db_name = os.getenv("DB_NAME")
conn = psycopg2.connect(
    dbname=db_name,
    user=db_user,
    password=db_password,
    host=db_host
)
cursor = conn.cursor()

inekler = [
    (1, 'Sarıkız', 'Holstein', 'Dişi ', datetime.date(2020, 5, 12), 482193),
    (2, 'Benekli', 'Simental', 'Dişi ', datetime.date(2021, 3, 15), 819204),
    (3, 'Karakız', 'Holstein', 'Dişi ', datetime.date(2019, 11, 20), 273849),
    (4, 'Alaca', 'Holstein', 'Dişi ', datetime.date(2022, 1, 8), 650192),
    (5, 'Gülizar', 'Montofon', 'Dişi ', datetime.date(2020, 9, 30), 192837),
    (6, 'Nazlı', 'Simental', 'Dişi ', datetime.date(2023, 2, 14), 564738),
    (7, 'Sultan', 'Holstein', 'Dişi ', datetime.date(2018, 7, 22), 920183),
    (8, 'Menekşe', 'Yerli Kara', 'Dişi ', datetime.date(2021, 6, 10), 374829),
    (9, 'Papatya', 'Simental', 'Dişi ', datetime.date(2022, 4, 5), 728190),
    (10, 'Ceylan', 'Jersey', 'Dişi ', datetime.date(2023, 8, 19), 182736),
    (11, 'Boncuk', 'Holstein', 'Dişi ', datetime.date(2020, 12, 1), 593827),
    (12, 'Yıldız', 'Montofon', 'Dişi ', datetime.date(2019, 5, 15), 402918),
    (13, 'Pamuk', 'Simental', 'Dişi ', datetime.date(2022, 10, 29), 837465),
    (14, 'Sümbül', 'Holstein', 'Dişi ', datetime.date(2021, 1, 17), 291038),
    (15, 'Kiraz', 'Montofon', 'Dişi ', datetime.date(2023, 3, 30), 647583),
    (16, 'Yazgülü', 'Simental', 'Dişi ', datetime.date(2020, 8, 14), 918273),
    (17, 'Zeytin', 'Yerli Kara', 'Dişi ', datetime.date(2021, 11, 11), 374658),
    (18, 'Bahar', 'Holstein', 'Dişi ', datetime.date(2022, 5, 23), 109283),
    (19, 'Yayla', 'Montofon', 'Dişi ', datetime.date(2019, 4, 4), 563728),
    (20, 'Döne', 'Simental', 'Dişi ', datetime.date(2023, 1, 10), 829102),
    (21, 'Maviş', 'Holstein', 'Dişi ', datetime.date(2020, 2, 28), 473829),
    (22, 'Kader', 'Holstein', 'Dişi ', datetime.date(2021, 9, 9), 657483),
    (23, 'Elmas', 'Simental', 'Dişi ', datetime.date(2018, 12, 12), 283746),
    (24, 'Çiçek', 'Montofon', 'Dişi ', datetime.date(2022, 7, 7), 910293),
    (25, 'Kadife', 'Yerli Kara', 'Dişi ', datetime.date(2019, 10, 3), 564739),
    (26, 'Fıstık', 'Jersey', 'Dişi ', datetime.date(2023, 6, 21), 182930),
    (27, 'Lale', 'Holstein', 'Dişi ', datetime.date(2021, 3, 18), 736452),
    (28, 'Güllü', 'Simental', 'Dişi ', datetime.date(2020, 11, 5), 392817),
    (29, 'Yağmur', 'Montofon', 'Dişi ', datetime.date(2022, 9, 14), 847563),
    (30, 'Şahane', 'Holstein', 'Dişi ', datetime.date(2023, 4, 25), 203948)
]


IRK_OZELLIKLERI = {
    'Holstein': {'sut_ort': 32, 'kilo_ort': 600},
    'Simental': {'sut_ort': 26, 'kilo_ort': 650},
    'Montofon': {'sut_ort': 22, 'kilo_ort': 550},
    'Jersey': {'sut_ort': 18, 'kilo_ort': 400},
    'Yerli Kara': {'sut_ort': 11, 'kilo_ort': 350},
}


def get_random_milk(irk):
   
    base = IRK_OZELLIKLERI.get(irk, {'sut_ort': 20})['sut_ort']
    daily_total = base * random.uniform(0.85, 1.15) 
    
  
    sabah_ratio = random.uniform(0.52, 0.58)
    sabah = daily_total * sabah_ratio
    aksam = daily_total * (1 - sabah_ratio)
    return round(sabah, 2), round(aksam, 2)

def get_new_weight(current_weight, irk):

    degisim_orani = random.uniform(0.98, 1.025) 
    return round(current_weight * degisim_orani, 2)


start_date = datetime.date(2025, 1, 1)
end_date = datetime.date(2025, 12, 31)
delta = datetime.timedelta(days=1)


current_weights = {}
for inek in inekler:
    irk = inek[2]
    base_weight = IRK_OZELLIKLERI.get(irk, {'kilo_ort': 500})['kilo_ort']

    current_weights[inek[0]] = base_weight * random.uniform(0.9, 1.1)


TARTIM_GUNU = 15 

print("SQL dosyası oluşturuluyor...")

with open('insert_data.sql', 'w', encoding='utf-8') as f:
    f.write("BEGIN;\n")
    
    current_day = start_date
    while current_day <= end_date:
        
       
        f.write(f"\n-- TARIH: {current_day} --\n")
        

        for inek in inekler:
            inek_id = inek[0]
            irk = inek[2]
            
            sabah, aksam = get_random_milk(irk)
            
            f.write(
                f"INSERT INTO sut (inek_id, sagim_tarihi, sabah_sagim, aksam_sagim) "
                f"VALUES ({inek_id}, '{current_day}', {sabah}, {aksam});\n"
            )
        
        
        if current_day.day == TARTIM_GUNU:
            f.write(f"\n-- TARTIM GÜNÜ ({current_day}) --\n")
            for inek in inekler:
                inek_id = inek[0]
                irk = inek[2]
                
              
                yeni_kilo = get_new_weight(current_weights[inek_id], irk)
                current_weights[inek_id] = yeni_kilo 
                
                f.write(
                    f"INSERT INTO kilo (inek_id, kilo, tartim_tarihi) "
                    f"VALUES ({inek_id}, {yeni_kilo}, '{current_day}');\n"
                )
        
        current_day += delta

    f.write("\nCOMMIT;")

print("Tamamlandı! Veriler 'insert_data.sql' dosyasına, takvim sırasına göre yazıldı.")


with open('insert_data.sql', 'r') as file:
    sql_script = file.read()

try:
    
    cursor.execute(sql_script)
    conn.commit()
    print("Veriler başarıyla yüklendi!")
except Exception as e:
    conn.rollback()
    print(f"Hata oluştu: {e}")
finally:
    cursor.close()
    conn.close()