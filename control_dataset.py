import os
import glob
import pandas as pd
import librosa
import tkinter as tk
from tkinter import filedialog
import warnings

# Librosa'dan gelen gereksiz uyarıları gizle
warnings.filterwarnings("ignore")

def detayli_veri_analizi():
    # Sadece klasör seçme penceresi açmak için arka planı gizle
    root = tk.Tk()
    root.withdraw()

    print("Lütfen kontrol edilecek Dataset ana klasörünü seçin...")
    dataset_path = filedialog.askdirectory(title="Dataset Klasörünü Seçin")

    if not dataset_path:
        print("Klasör seçimi iptal edildi.")
        return

    print("\n" + "="*70)
    print(f"🔍 DETAYLI VERİ SETİ KONTROL RAPORU BAŞLADI")
    print(f"Hedef Klasör: {dataset_path}")
    print("="*70)

    # 1. Excel Dosyalarını Bul [cite: 53]
    excel_files = glob.glob(f"{dataset_path}/**/*.xlsx", recursive=True)
    if not excel_files:
        print("❌ CRITICAL HATA: Seçilen klasörde hiçbir .xlsx dosyası bulunamadı!")
        return

    toplam_ses_kaydi = 0
    hatalar = []

    # Ana kodun beklediği kesin sütun isimleri [cite: 53]
    beklenen_sutun_dosya = 'File_Path'
    beklenen_sutun_cinsiyet = 'Gender'

    # 2. Her Excel dosyasını tek tek incele
    for excel_file in excel_files:
        excel_isim = os.path.basename(excel_file)
        klasor_yolu = os.path.dirname(excel_file)
        print(f"\n📄 İNCELENİYOR: {excel_isim} (Yol: {klasor_yolu})")

        try:
            # Excel'i oku
            df = pd.read_excel(excel_file)
            kolonlar = df.columns.tolist()
            
            # Sütun isimlerini özellikle tırnak içinde yazdırıyoruz ki gizli boşluklar (örn: 'File_Path ') belli olsun
            print(f"   ✓ Tespit Edilen Gerçek Sütun Başlıkları:")
            for c in kolonlar:
                print(f"      -> '{c}'")

            # A. SÜTUN İSMİ KONTROLLERİ
            if beklenen_sutun_dosya not in kolonlar:
                hatalar.append(f"[EKSİK SÜTUN] '{excel_isim}' içinde tam olarak '{beklenen_sutun_dosya}' adında bir başlık yok.")
                
                # Kullanıcıya ne yazması gerektiğini önermek için benzer kelimeleri ara
                olasi_dosyalar = [c for c in kolonlar if 'file' in str(c).lower() or 'dosya' in str(c).lower() or 'path' in str(c).lower()]
                if olasi_dosyalar:
                    hatalar.append(f"  ↳ TAVSİYE: Sanırım '{beklenen_sutun_dosya}' yerine şunu kullanmışsınız: '{olasi_dosyalar[0]}'. Ana koddaki ismi buna göre değiştirin.")
                
                continue # Dosya yolu sütunu yoksa, içindeki sesleri test edemeyiz, diğer dosyaya geç.

            if beklenen_sutun_cinsiyet not in kolonlar:
                hatalar.append(f"[EKSİK SÜTUN] '{excel_isim}' içinde '{beklenen_sutun_cinsiyet}' sütunu yok. Doğruluk oranı hesaplanamayacak.")

            # B. SATIR VE SES DOSYASI KONTROLLERİ
            for index, row in df.iterrows():
                # Excel satırı (örn: 2. satır)
                gercek_satir_no = index + 2 
                dosya_adi = str(row[beklenen_sutun_dosya]).strip()

                if pd.isna(row[beklenen_sutun_dosya]) or dosya_adi == 'nan' or dosya_adi == '':
                    hatalar.append(f"[BOŞ HÜCRE] '{excel_isim}' Satır {gercek_satir_no}: Dosya adı hücresi boş!")
                    continue

                # Dosya yolunu oluştur (Excel dosyası ile aynı klasörde olduğunu varsayarak)
                if os.path.isabs(dosya_adi):
                    tam_dosya_yolu = dosya_adi
                else:
                    tam_dosya_yolu = os.path.join(klasor_yolu, dosya_adi)

                toplam_ses_kaydi += 1

                # 1. Ses dosyası fiziksel olarak var mı?
                if not os.path.exists(tam_dosya_yolu):
                    hatalar.append(f"[KAYIP DOSYA] '{excel_isim}' (Satır {gercek_satir_no}) -> '{dosya_adi}' klasörde fiziksel olarak yok!")
                else:
                    # 2. Ses dosyası bozuk mu? (Sadece ilk 0.1 saniyesini yükleyerek hızlıca test et)
                    try:
                        _ = librosa.load(tam_dosya_yolu, sr=None, duration=0.1)
                    except Exception as e:
                        hata_nedeni = str(e).splitlines()[0]
                        hatalar.append(f"[BOZUK DOSYA] '{dosya_adi}' okunamıyor! Hata: {hata_nedeni}")

        except Exception as e:
            hatalar.append(f"[EXCEL OKUMA HATASI] '{excel_isim}' bozuk veya açılamıyor. Detay: {str(e)}")

    # 3. GENEL SONUÇ RAPORUNU YAZDIR
    print("\n" + "="*70)
    print("📋 KESİN TEŞHİS VE KONTROL ÖZETİ")
    print("="*70)
    print(f"Taranan Excel Dosyası Sayısı: {len(excel_files)}")
    print(f"İncelenen Ses Dosyası Kaydı: {toplam_ses_kaydi}")

    if not hatalar:
        print("\n✅ MÜKEMMEL! Veri setinin dosya yapısında, sütun başlıklarında veya ses dosyalarında HİÇBİR HATA YOK.")
        print("Ana projeyi sorunsuz bir şekilde çalıştırabilirsiniz.")
    else:
        print(f"\n❌ DİKKAT! TOPLAM {len(hatalar)} ADET KRİTİK HATA TESPİT EDİLDİ:\n")
        for i, hata in enumerate(hatalar, 1):
            print(f"{i}. {hata}")
            
    print("\n======================================================================")

if __name__ == "__main__":
    detayli_veri_analizi()