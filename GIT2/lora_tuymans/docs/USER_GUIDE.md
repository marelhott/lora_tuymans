# 📖 Uživatelský Manuál

## Přehled

Tento manuál vás provede kompletním používáním aplikace LoRA Tuymans Style Transfer od základního nastavení po pokročilé techniky.

## 📋 Obsah

- [🚀 První spuštění](#-první-spuštění)
- [🎨 Základní použití](#-základní-použití)
- [⚙️ Nastavení parametrů](#️-nastavení-parametrů)
- [🖼️ Práce s modely](#️-práce-s-modely)
- [🔧 Pokročilé funkce](#-pokročilé-funkce)
- [💡 Tipy a triky](#-tipy-a-triky)
- [❓ FAQ](#-faq)
- [🐛 Řešení problémů](#-řešení-problémů)

---

## 🚀 První spuštění

### Přístup k aplikaci

1. **Lokální spuštění:**
   - Otevřete webový prohlížeč
   - Přejděte na `http://localhost:8501`

2. **Docker/Cloud:**
   - Použijte IP adresu serveru: `http://YOUR-SERVER-IP:8501`

### První pohled na rozhraní

```
┌─────────────────────────────────────────────────────────────┐
│                    LoRA Tuymans Style Transfer             │
├─────────────┬─────────────────────────┬─────────────────────┤
│   LEVÝ      │       STŘEDNÍ           │      PRAVÝ          │
│  SIDEBAR    │      SLOUPEC            │     SIDEBAR         │
│             │                         │                     │
│ • Model     │   [GENEROVAT]           │ • Browse files      │
│ • Parametry │                         │ • LoRA modely       │
│ • Batch     │   Progress bar          │ • Full modely       │
│             │                         │                     │
│             │   Výsledné obrázky      │                     │
└─────────────┴─────────────────────────┴─────────────────────┘
```

### Kontrola systému

**Indikátory v aplikaci:**
- 🟢 **GPU dostupná** - Zelená ikona
- 🟡 **CPU režim** - Žlutá ikona
- 🔴 **Chyba** - Červená ikona

---

## 🎨 Základní použití

### Krok 1: Nahrání obrázku

1. **Klikněte na "Browse files"** v pravém sidebaru
2. **Vyberte obrázek** z vašeho počítače
   - Podporované formáty: PNG, JPG, JPEG
   - Maximální velikost: 10GB
   - Doporučené rozlišení: 512x512 až 2048x2048px

3. **Náhled obrázku** se zobrazí pod tlačítkem

**💡 Tip:** Pro nejlepší výsledky použijte obrázky s:
- Dobrým kontrastem
- Jasně definovanými objekty
- Rozlišením alespoň 512x512px

### Krok 2: Výběr modelu

#### LoRA modely (Doporučené)
1. **Umístěte .safetensors soubory** do složky `data/loras/`
2. **Klikněte na kroužek** vedle názvu modelu
3. **Kroužek zčerná** = model je vybrán

#### Full modely
1. **Umístěte .safetensors soubory** do složky `data/models/`
2. **Klikněte na kroužek** vedle názvu modelu
3. **Pouze jeden model** může být aktivní současně

**📁 Struktura složek:**
```
data/
├── loras/
│   ├── tuymans_style_v1.safetensors
│   ├── tuymans_portraits.safetensors
│   └── tuymans_landscapes.safetensors
└── models/
    ├── realistic_vision_v5.safetensors
    └── deliberate_v2.safetensors
```

### Krok 3: Základní parametry

**Doporučené nastavení pro začátečníky:**
- **Strength:** 0.8 (80% síla stylu)
- **CFG Scale:** 7.5 (vyvážené řízení)
- **Steps:** 20 (dobrý poměr kvalita/rychlost)
- **Clip Skip:** 2 (standardní)
- **Batch Count:** 1 (jedna varianta)

### Krok 4: Generování

1. **Klikněte "Generovat"**
2. **Sledujte progress bar:**
   - MODEL_LOAD (0-50%)
   - IMG_GENERATE (50-100%)
3. **Výsledky se zobrazí automaticky**

### Krok 5: Prohlížení výsledků

- **Klikněte na obrázek** pro lightbox
- **Dvojklik** pro zavření lightboxu
- **Obrázky jsou automaticky uloženy** v session

---

## ⚙️ Nastavení parametrů

### Strength (Síla stylu)

**Rozsah:** 0.1 - 1.0

| Hodnota | Efekt | Použití |
|---------|-------|----------|
| 0.1-0.3 | Jemné změny | Zachování původního stylu |
| 0.4-0.6 | Mírné změny | Subtilní stylizace |
| 0.7-0.8 | Výrazné změny | Standardní stylový přenos |
| 0.9-1.0 | Dramatické změny | Kompletní transformace |

**💡 Doporučení:**
- **Portréty:** 0.6-0.8
- **Krajiny:** 0.7-0.9
- **Abstraktní:** 0.8-1.0

### CFG Scale (Guidance Scale)

**Rozsah:** 1.0 - 20.0

| Hodnota | Efekt | Kvalita |
|---------|-------|----------|
| 1-3 | Velmi volné | Kreativní, neprediktabilní |
| 4-7 | Vyvážené | Dobrý kompromis |
| 8-12 | Přísné | Konzistentní, kontrolované |
| 13-20 | Velmi přísné | Může způsobit artefakty |

**💡 Doporučení:**
- **Experimenty:** 5-7
- **Produkce:** 7-9
- **Detailní práce:** 9-12

### Steps (Inference kroky)

**Rozsah:** 10 - 50

| Hodnota | Čas | Kvalita | Použití |
|---------|-----|---------|----------|
| 10-15 | Rychlé | Základní | Náhledy, testy |
| 16-25 | Střední | Dobrá | Standardní práce |
| 26-35 | Pomalé | Vysoká | Finální výstupy |
| 36-50 | Velmi pomalé | Maximální | Profesionální práce |

**💡 Doporučení:**
- **Testování:** 15-20
- **Produkce:** 20-30
- **Vysoká kvalita:** 30-40

### Clip Skip

**Rozsah:** 1 - 2

| Hodnota | Efekt |
|---------|-------|
| 1 | Více detailní, přesnější |
| 2 | Více kreativní, stylizované |

**💡 Doporučení:**
- **Realistické obrázky:** 1
- **Stylizované obrázky:** 2

### Batch Count

**Rozsah:** 1 - 3

- **1 obrázek:** Rychlé, úspora paměti
- **2 obrázky:** Dobrý kompromis
- **3 obrázky:** Více variant, pomalejší

---

## 🖼️ Práce s modely

### Typy modelů

#### LoRA modely
**Výhody:**
- ✅ Rychlé načítání
- ✅ Menší velikost (50-200MB)
- ✅ Kombinovatelné
- ✅ Specifické styly

**Nevýhody:**
- ❌ Vyžadují base model
- ❌ Omezené možnosti

#### Full modely
**Výhody:**
- ✅ Kompletní funkcionalita
- ✅ Nezávislé
- ✅ Vysoká kvalita

**Nevýhody:**
- ❌ Velká velikost (2-7GB)
- ❌ Pomalé načítání
- ❌ Více paměti

### Získání modelů

#### Oficiální zdroje
1. **HuggingFace Hub**
   - `https://huggingface.co/models`
   - Filtr: "safetensors"

2. **Civitai**
   - `https://civitai.com`
   - Komunita modelů

3. **GitHub repozitáře**
   - Oficiální releases

#### Instalace modelů

```bash
# LoRA modely
wget -O data/loras/tuymans_style.safetensors "MODEL_URL"

# Full modely
wget -O data/models/realistic_vision.safetensors "MODEL_URL"

# Ověření
ls -la data/loras/
ls -la data/models/
```

### Doporučené modely

#### Pro Tuymans styl
- **tuymans_portraits_v2.safetensors** - Portréty
- **tuymans_landscapes_v1.safetensors** - Krajiny
- **tuymans_abstract_v3.safetensors** - Abstraktní

#### Base modely
- **Realistic Vision v5.0** - Realistické obrázky
- **Deliberate v2** - Umělecké styly
- **DreamShaper v8** - Univerzální

---

## 🔧 Pokročilé funkce

### Random Seed

**Aktivace:**
1. ☑️ Zaškrtněte "Random Seed"
2. Zadejte číslo (0-2147483647)

**Použití:**
- **Reprodukovatelnost:** Stejný seed = stejný výsledek
- **Experimenty:** Různé seedy = různé varianty
- **Srovnání:** Testování parametrů se stejným seedem

**💡 Tipy:**
- Oblíbené seedy: 42, 123, 777, 1337
- Uložte si seedy úspěšných generování
- Pro náhodnost nechte nezaškrtnuté

### Variance Seed

**Aktivace:**
1. ☑️ Zaškrtněte "Variance Seed"
2. Zadejte variance seed
3. Nastavte variance strength (0.0-1.0)

**Parametry:**
- **Variance Seed:** Seed pro variace
- **Variance Strength:** Síla variací
  - 0.1 = jemné změny
  - 0.5 = střední změny
  - 1.0 = výrazné změny

**Použití:**
- Generování podobných, ale různých variant
- Jemné úpravy úspěšných výsledků

### Upscaling

**Aktivace:**
1. ☑️ Zaškrtněte "Upscaling"
2. Vyberte faktor (2x nebo 4x)

**Faktory:**
- **2x:** 512x512 → 1024x1024
- **4x:** 512x512 → 2048x2048

**⚠️ Upozornění:**
- Výrazně zvyšuje čas generování
- Vyžaduje více VRAM
- Kvalita závisí na původním obrázku

### Kombinace parametrů

**Kreativní experimenty:**
```
Strength: 0.9
CFG Scale: 5.0
Steps: 30
Variance: 0.3
```

**Konzistentní výsledky:**
```
Strength: 0.7
CFG Scale: 8.0
Steps: 25
Seed: 42
```

**Vysoká kvalita:**
```
Strength: 0.8
CFG Scale: 7.5
Steps: 40
Upscaling: 2x
```

---

## 💡 Tipy a triky

### Optimalizace výkonu

#### GPU optimalizace
```bash
# Environment variables
ENABLE_ATTENTION_SLICING=true
ENABLE_CPU_OFFLOAD=false  # Pokud máte dostatek VRAM
MAX_MEMORY_GB=16
```

#### Batch processing
- **Malá VRAM (8GB):** Batch Count = 1
- **Střední VRAM (12GB):** Batch Count = 2
- **Velká VRAM (16GB+):** Batch Count = 3

### Kvalita obrázků

#### Příprava vstupního obrázku
1. **Rozlišení:** Ideálně 512x512, 768x768, nebo 1024x1024
2. **Formát:** PNG pro nejlepší kvalitu
3. **Kontrast:** Upravte kontrast před nahráním
4. **Kompozice:** Jasně definované objekty

#### Optimální parametry podle typu

**Portréty:**
```
Strength: 0.6-0.8
CFG Scale: 7-9
Steps: 25-35
Clip Skip: 1
```

**Krajiny:**
```
Strength: 0.7-0.9
CFG Scale: 6-8
Steps: 20-30
Clip Skip: 2
```

**Abstraktní umění:**
```
Strength: 0.8-1.0
CFG Scale: 5-7
Steps: 30-40
Clip Skip: 2
```

### Workflow doporučení

#### 1. Testovací fáze
```
1. Nahrajte obrázek
2. Vyberte model
3. Nastavte: Strength 0.8, CFG 7.5, Steps 15
4. Generujte 1 obrázek
5. Vyhodnoťte výsledek
```

#### 2. Optimalizační fáze
```
1. Upravte parametry podle výsledku
2. Zvyšte Steps na 25-30
3. Experimentujte se Strength
4. Testujte různé seedy
```

#### 3. Finální fáze
```
1. Optimální parametry
2. Steps 30-40
3. Batch Count 2-3
4. Volitelně Upscaling
```

### Řešení častých problémů

#### Obrázek je příliš změněný
- ⬇️ Snižte Strength (0.5-0.7)
- ⬇️ Snižte CFG Scale (5-6)
- ⬆️ Zvyšte Steps (25-30)

#### Obrázek je málo změněný
- ⬆️ Zvyšte Strength (0.8-0.9)
- ⬆️ Zvyšte CFG Scale (8-10)
- Zkuste jiný model

#### Artefakty a chyby
- ⬇️ Snižte CFG Scale (6-7)
- ⬆️ Zvyšte Steps (30-40)
- Zkuste Clip Skip 1

#### Pomalé generování
- ⬇️ Snižte Steps (15-20)
- ⬇️ Snižte Batch Count (1)
- Vypněte Upscaling

---

## ❓ FAQ

### Obecné otázky

**Q: Jak dlouho trvá generování?**
A: Závisí na hardware:
- RTX 4090: 8-15 sekund
- RTX 3080: 15-25 sekund
- CPU: 3-8 minut

**Q: Jaké formáty obrázků jsou podporované?**
A: PNG, JPG, JPEG pro vstup. Výstup je vždy PNG.

**Q: Můžu používat vlastní modely?**
A: Ano, umístěte .safetensors soubory do příslušných složek.

**Q: Kolik paměti potřebuji?**
A: Minimum 8GB RAM, doporučeno 16GB+. Pro GPU minimum 6GB VRAM.

### Technické otázky

**Q: Proč se aplikace přepne na CPU?**
A: Možné příčiny:
- Nedostatek VRAM
- CUDA chyba
- Chybějící NVIDIA drivers

**Q: Můžu spustit více instancí?**
A: Ano, ale každá potřebuje vlastní port a GPU paměť.

**Q: Jak zálohovat modely?**
A: Zkopírujte složky `data/loras/` a `data/models/`.

**Q: Podporuje aplikace AMD GPU?**
A: Ne, pouze NVIDIA GPU s CUDA podporou.

### Řešení problémů

**Q: "CUDA out of memory" chyba**
A: 
1. Snižte Batch Count na 1
2. Povolte CPU offload
3. Restartujte aplikaci

**Q: Model se nenačte**
A:
1. Zkontrolujte cestu k souboru
2. Ověřte formát (.safetensors)
3. Zkontrolujte oprávnění

**Q: Aplikace se nespustí**
A:
1. Zkontrolujte dependencies: `pip check`
2. Aktualizujte Streamlit: `pip install -U streamlit`
3. Zkontrolujte port: `netstat -tulpn | grep 8501`

---

## 🐛 Řešení problémů

### Diagnostika

#### Kontrola systému
```bash
# GPU status
nvidia-smi

# Python environment
python --version
pip list | grep torch
pip list | grep streamlit

# Disk space
df -h

# Memory
free -h
```

#### Kontrola aplikace
```bash
# Logs
docker logs lora_tuymans

# Health check
curl http://localhost:8501/_stcore/health

# Process
ps aux | grep streamlit
```

### Časté chyby

#### "ModuleNotFoundError"
```bash
# Řešení
pip install -r requirements.txt
# nebo
pip install torch torchvision diffusers streamlit
```

#### "Permission denied"
```bash
# Řešení
sudo chown -R $USER:$USER data/
chmod 755 data/loras/ data/models/
```

#### "Port 8501 already in use"
```bash
# Najít proces
sudo lsof -i :8501

# Ukončit
sudo kill -9 PID

# Nebo použít jiný port
streamlit run app.py --server.port=8502
```

#### "CUDA initialization failed"
```bash
# Restart NVIDIA services
sudo systemctl restart nvidia-docker

# Nebo vynutit CPU
export FORCE_CPU=true
streamlit run app.py
```

### Performance problémy

#### Pomalé načítání modelů
- Použijte SSD disk
- Zvyšte RAM
- Zkontrolujte síťové připojení (pro download)

#### Vysoké využití paměti
```bash
# Monitoring
watch -n 1 'free -h && nvidia-smi'

# Optimalizace
ENABLE_ATTENTION_SLICING=true
ENABLE_CPU_OFFLOAD=true
MAX_MEMORY_GB=8
```

#### Nestabilní výsledky
- Použijte fixed seed
- Zkontrolujte teplotu GPU
- Aktualizujte drivers

### Kontakt a podpora

**📧 Email:** [your-email@example.com](mailto:your-email@example.com)  
**🐛 Issues:** [GitHub Issues](https://github.com/your-username/lora_tuymans/issues)  
**💬 Diskuze:** [GitHub Discussions](https://github.com/your-username/lora_tuymans/discussions)  
**📖 Dokumentace:** [README.md](../README.md)

---

**🎉 Gratulujeme! Nyní ovládáte LoRA Tuymans Style Transfer aplikaci.**

**Užijte si tvorbu uměleckých děl v stylu Luca Tuymanse! 🎨**