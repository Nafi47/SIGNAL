import os
import glob
import pandas as pd
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tkinter as tk
from tkinter import filedialog, messagebox
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. SİNYAL İŞLEME FONKSİYONLARI (YÜKSEK DOĞRULUK)
# ==========================================
def extract_voiced_frames(audio, sr, frame_ms=25):
    frame_length = int(sr * (frame_ms / 1000.0))
    hop_length = int(frame_length / 2) 
    
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=frame_length, hop_length=hop_length)[0]
    energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    
    energy_threshold = np.percentile(energy, 75) * 0.15 
    zcr_threshold = np.mean(zcr) * 1.5
    
    voiced_indices = np.where((energy > energy_threshold) & (zcr < zcr_threshold))[0]
    
    voiced_frames = []
    for idx in voiced_indices:
        start = idx * hop_length
        end = start + frame_length
        if end <= len(audio):
            voiced_frames.append(audio[start:end])
            
    avg_zcr = np.mean(zcr[voiced_indices]) if len(voiced_indices) > 0 else 0
    avg_energy = np.mean(energy[voiced_indices]) if len(voiced_indices) > 0 else 0
            
    return voiced_frames, avg_zcr, avg_energy

def compute_autocorrelation_pitch(frame, sr):
    windowed_frame = frame * np.hanning(len(frame))
    autocorr = np.correlate(windowed_frame, windowed_frame, mode='full')
    autocorr = autocorr[len(autocorr)//2:] 
    
    min_lag = int(sr / 400)
    max_lag = int(sr / 70)
    
    autocorr_window = autocorr[min_lag:max_lag]
    if len(autocorr_window) == 0: return 0
        
    peak_lag = np.argmax(autocorr_window) + min_lag
    f0 = sr / peak_lag
    return f0

def plot_autocorr_vs_fft(frame, sr):
    windowed_frame = frame * np.hanning(len(frame))
    autocorr = np.correlate(windowed_frame, windowed_frame, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    
    fft_spectrum = np.abs(np.fft.rfft(windowed_frame))
    freqs = np.fft.rfftfreq(len(windowed_frame), 1/sr)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(autocorr)
    plt.title("Otokorelasyon (Zaman Tanım Kümesi)")
    plt.xlabel("Gecikme (Lag)")
    plt.ylabel("Genlik")
    
    plt.subplot(1, 2, 2)
    plt.plot(freqs, fft_spectrum)
    plt.title("FFT Spektrumu (Frekans Tanım Kümesi)")
    plt.xlabel("Frekans (Hz)")
    plt.ylabel("Büyüklük")
    plt.xlim(0, 1000) 
    
    plt.tight_layout()
    plt.show()

def classify_gender(f0_val):
    if f0_val == 0: return "Bilinmiyor"
    elif f0_val < 175: return "Male"
    elif f0_val >= 175 and f0_val < 255: return "Woman"
    else: return "Child"

def standardize_label(raw_label):
    """Hem Türkçe hem İngilizce etiketleri ayrım yapmaksızın tek tipe dönüştürür."""
    val = str(raw_label).strip().lower()
    if val in ['k', 'kadın', 'kadin', 'female', 'woman', 'f', 'w']: return 'Woman'
    if val in ['e', 'erkek', 'male', 'm']: return 'Male'
    if val in ['c', 'çocuk', 'cocuk', 'child']: return 'Child'
    return "Bilinmiyor"

def process_dataset(dataset_path):
    excel_files = glob.glob(f"{dataset_path}/**/*.xlsx", recursive=True)
    if not excel_files:
        raise FileNotFoundError("Klasörde hiçbir Excel (.xlsx) dosyası bulunamadı.")
        
    results = []
    
    for excel_file in excel_files:
        dosya_ismi = os.path.basename(excel_file)
        if dosya_ismi.startswith('~$') or dosya_ismi.startswith('._'): continue
            
        try:
            df = pd.read_excel(excel_file, engine='openpyxl')
        except Exception: continue

        kolonlar = df.columns.tolist()
        
        # --- ESNEK SÜTUN BULUCU: Türkçe/İngilizce ayrımı yapmaz ---
        dosya_kolonu = None
        for c in kolonlar:
            c_temiz = str(c).lower().replace("_", " ").strip()
            if "dosya" in c_temiz or "file" in c_temiz or "path" in c_temiz:
                dosya_kolonu = c
                break

        cinsiyet_kolonu = None
        for c in kolonlar:
            c_temiz = str(c).lower().replace("_", " ").strip()
            if "cinsiyet" in c_temiz or "gender" in c_temiz or "sex" in c_temiz:
                cinsiyet_kolonu = c
                break
        
        if not dosya_kolonu: continue
            
        klasor_yolu = os.path.dirname(excel_file)
        
        for index, row in df.iterrows():
            ham_dosya_adi = str(row[dosya_kolonu]).strip()
            if pd.isna(row[dosya_kolonu]) or ham_dosya_adi == 'nan' or ham_dosya_adi == '': continue
                
            if not ham_dosya_adi.lower().endswith('.wav'): ham_dosya_adi += '.wav'
                
            tam_dosya_yolu = os.path.join(klasor_yolu, ham_dosya_adi)
            
            gercek_etiket = standardize_label(row[cinsiyet_kolonu]) if cinsiyet_kolonu else "Bilinmiyor"
            
            if not os.path.exists(tam_dosya_yolu): continue
                
            try:
                audio, sr = librosa.load(tam_dosya_yolu, sr=None)
                voiced_frames, avg_zcr, avg_energy = extract_voiced_frames(audio, sr)
                
                f0_values = [compute_autocorrelation_pitch(f, sr) for f in voiced_frames if compute_autocorrelation_pitch(f, sr) > 0]
                
                if len(f0_values) > 2:
                    q1, q3 = np.percentile(f0_values, [25, 75])
                    iqr = q3 - q1
                    alt_sinir = q1 - (1.5 * iqr)
                    ust_sinir = q3 + (1.5 * iqr)
                    temiz_f0_degerleri = [f for f in f0_values if alt_sinir <= f <= ust_sinir]
                    hesaplanan_f0 = np.median(temiz_f0_degerleri) if len(temiz_f0_degerleri) > 0 else np.median(f0_values)
                else:
                    hesaplanan_f0 = np.median(f0_values) if len(f0_values) > 0 else 0
                
                predicted_label = classify_gender(hesaplanan_f0)
                
                results.append({
                    'File_Path': tam_dosya_yolu,
                    'True_Label': gercek_etiket,
                    'Predicted_Label': predicted_label,
                    'Avg_F0': hesaplanan_f0,
                    'Avg_ZCR': avg_zcr,
                    'Avg_Energy': avg_energy
                })
            except Exception: continue
                
    if not results: raise ValueError("Hiçbir ses dosyası işlenemedi.")
        
    return pd.DataFrame(results)

# ==========================================
# 2. MASAÜSTÜ ARAYÜZÜ (TKINTER) SINIFI
# ==========================================
class SesSiniflandirmaUygulamasi:
    def __init__(self, root):
        self.root = root
        self.root.title("Ses Analizi ve Sınıflandırma - Proje Arayüzü")
        self.root.geometry("600x450")
        self.root.configure(padx=20, pady=20)

        self.lbl_baslik = tk.Label(root, text="Kural Tabanlı Cinsiyet Sınıflandırıcı", font=("Arial", 16, "bold"))
        self.lbl_baslik.pack(pady=10)

        self.btn_dosya_sec = tk.Button(root, text="1. Canlı Demo: Ses Sınıflandır ve Grafiği Çiz (.wav)", command=self.tek_dosya_analiz_et, width=45, height=2, bg="lightblue")
        self.btn_dosya_sec.pack(pady=10)

        self.lbl_sonuc = tk.Label(root, text="Seçilen Dosya Sonucu:\nBekleniyor...", font=("Arial", 11), justify=tk.LEFT, relief=tk.SUNKEN, width=55, height=4)
        self.lbl_sonuc.pack(pady=10)

        tk.Frame(root, height=2, bd=1, relief=tk.SUNKEN).pack(fill=tk.X, pady=10)

        self.btn_veri_seti = tk.Button(root, text="2. Veri Setini Test Et & Karmaşıklık Matrisi Çiz (Klasör Seç)", command=self.veri_seti_analiz_et, width=45, height=2, bg="lightgreen")
        self.btn_veri_seti.pack(pady=10)

        self.lbl_basari = tk.Label(root, text="Genel Veri Seti Başarısı: Bekleniyor...", font=("Arial", 12, "bold"))
        self.lbl_basari.pack(pady=10)

    def tek_dosya_analiz_et(self):
        dosya_yolu = filedialog.askopenfilename(title="Bir WAV dosyası seçin", filetypes=[("Audio Files", "*.wav")])
        if not dosya_yolu: return

        self.lbl_sonuc.config(text="Analiz ediliyor... Lütfen bekleyin.")
        self.root.update()

        try:
            audio, sr = librosa.load(dosya_yolu, sr=None)
            voiced_frames, avg_zcr, avg_energy = extract_voiced_frames(audio, sr)
            
            f0_values = [compute_autocorrelation_pitch(f, sr) for f in voiced_frames if compute_autocorrelation_pitch(f, sr) > 0]
            
            if len(f0_values) > 2:
                q1, q3 = np.percentile(f0_values, [25, 75])
                iqr = q3 - q1
                alt_sinir = q1 - (1.5 * iqr)
                ust_sinir = q3 + (1.5 * iqr)
                temiz_f0_degerleri = [f for f in f0_values if alt_sinir <= f <= ust_sinir]
                hesaplanan_f0 = np.median(temiz_f0_degerleri) if len(temiz_f0_degerleri) > 0 else np.median(f0_values)
            else:
                hesaplanan_f0 = np.median(f0_values) if len(f0_values) > 0 else 0
                
            tahmin = classify_gender(hesaplanan_f0)

            sonuc_metni = (f"Dosya: {dosya_yolu.split('/')[-1]}\nTahmin Edilen Sınıf: {tahmin}\nMedyan F0: {hesaplanan_f0:.2f} Hz | ZCR: {avg_zcr:.4f}")
            self.lbl_sonuc.config(text=sonuc_metni)

            if len(voiced_frames) > 0:
                plot_autocorr_vs_fft(voiced_frames[0], sr)
            else:
                messagebox.showwarning("Uyarı", "Ses analiz edildi ancak grafiği çizilecek 'sesli bir bölge (voiced frame)' bulunamadı.")

        except Exception as e:
            self.lbl_sonuc.config(text="Hata oluştu.")
            messagebox.showerror("Hata", f"Dosya işlenirken bir hata oluştu:\n{str(e)}")

    def veri_seti_analiz_et(self):
        klasor_yolu = filedialog.askdirectory(title="Dataset Klasörünü Seçin")
        if not klasor_yolu: return

        self.lbl_basari.config(text="Tüm veri seti işleniyor... (Bu biraz sürebilir)")
        self.root.update()

        try:
            df_sonuclar = process_dataset(klasor_yolu)

            df_sonuclar['True_Label_Lower'] = df_sonuclar['True_Label'].str.lower()
            df_sonuclar['Predicted_Label_Lower'] = df_sonuclar['Predicted_Label'].str.lower()
            
            dogru_tahminler = df_sonuclar[df_sonuclar['True_Label_Lower'] == df_sonuclar['Predicted_Label_Lower']]
            basari_orani = (len(dogru_tahminler) / len(df_sonuclar)) * 100
            self.lbl_basari.config(text=f"Genel Veri Seti Başarısı: %{basari_orani:.2f}\n({len(dogru_tahminler)} / {len(df_sonuclar)} doğru tahmin)")
            
            gercek = df_sonuclar['True_Label'].tolist()
            tahmin = df_sonuclar['Predicted_Label'].tolist()
            etiketler = ['Male', 'Woman', 'Child']
            
            cm = confusion_matrix(gercek, tahmin, labels=etiketler)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=etiketler)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            disp.plot(ax=ax, cmap=plt.cm.Blues)
            plt.title("Karmaşıklık Matrisi (Confusion Matrix)")
            plt.show()

        except Exception as e:
            self.lbl_basari.config(text="Genel Veri Seti Başarısı: HATA OLUŞTU")
            messagebox.showerror("Veri Seti Hatası", str(e))

# ==========================================
# 3. UYGULAMAYI BAŞLATMA
# ==========================================
if __name__ == "__main__":
    root = tk.Tk()
    uygulama = SesSiniflandirmaUygulamasi(root)
    root.mainloop()