# 🤝 Contributing Guide

## Přehled

Děkujeme za váš zájem přispět do projektu LoRA Tuymans Style Transfer! Tento guide vás provede procesem přispívání.

## 📋 Obsah

- [🚀 Rychlý start](#-rychlý-start)
- [🔧 Development setup](#-development-setup)
- [📝 Coding standards](#-coding-standards)
- [🧪 Testing](#-testing)
- [📖 Dokumentace](#-dokumentace)
- [🔄 Pull Request proces](#-pull-request-proces)
- [🐛 Bug reporting](#-bug-reporting)
- [💡 Feature requests](#-feature-requests)
- [👥 Community](#-community)

---

## 🚀 Rychlý start

### Typy příspěvků

Vítáme následující typy příspěvků:

- 🐛 **Bug fixes** - Opravy chyb
- ✨ **Features** - Nové funkce
- 📖 **Documentation** - Vylepšení dokumentace
- 🎨 **UI/UX** - Vylepšení uživatelského rozhraní
- ⚡ **Performance** - Optimalizace výkonu
- 🧪 **Tests** - Přidání nebo vylepšení testů
- 🔧 **Refactoring** - Refaktoring kódu

### Před začátkem

1. **Zkontrolujte existující issues** - Možná už někdo pracuje na podobném problému
2. **Diskutujte velké změny** - Otevřete issue pro diskuzi před implementací
3. **Přečtěte si tento guide** - Seznamte se s našimi standardy

---

## 🔧 Development Setup

### Požadavky

- Python 3.10+
- Git
- NVIDIA GPU (doporučeno)
- 16GB+ RAM

### Fork a clone

```bash
# 1. Fork repozitáře na GitHubu
# 2. Clone vašeho forku
git clone https://github.com/YOUR-USERNAME/lora_tuymans.git
cd lora_tuymans

# 3. Přidání upstream remote
git remote add upstream https://github.com/original-owner/lora_tuymans.git
```

### Environment setup

```bash
# Virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Development dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Pre-commit hooks

```bash
# Instalace pre-commit
pip install pre-commit

# Setup hooks
pre-commit install

# Test hooks
pre-commit run --all-files
```

### Development dependencies

**`requirements-dev.txt`**
```
# Testing
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-mock>=3.10.0

# Code quality
black>=23.0.0
isort>=5.12.0
flake8>=6.0.0
mypy>=1.0.0

# Pre-commit
pre-commit>=3.0.0

# Documentation
mkdocs>=1.4.0
mkdocs-material>=9.0.0

# Development tools
ipython>=8.0.0
jupyter>=1.0.0
```

---

## 📝 Coding Standards

### Python Style Guide

Dodržujeme **PEP 8** s následujícími specifiky:

#### Formatting

```python
# Black formatter
black --line-length 88 --target-version py310 .

# Import sorting
isort --profile black .
```

#### Type Hints

```python
# Vždy používejte type hints
from typing import List, Dict, Optional, Union
from PIL import Image

def apply_style(
    input_image: Image.Image,
    model_path: str,
    strength: float = 0.8,
    progress_callback: Optional[callable] = None
) -> List[Image.Image]:
    """Apply style transfer to input image.
    
    Args:
        input_image: Input PIL Image
        model_path: Path to model file
        strength: Style strength (0.0-1.0)
        progress_callback: Optional progress callback
        
    Returns:
        List of generated images
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If CUDA out of memory
    """
    pass
```

#### Docstrings

```python
# Google style docstrings
def function_name(param1: str, param2: int) -> bool:
    """Brief description of function.
    
    Longer description if needed. Explain the purpose,
    behavior, and any important details.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is invalid
        RuntimeError: When operation fails
        
    Example:
        >>> result = function_name("test", 42)
        >>> print(result)
        True
    """
    pass
```

#### Error Handling

```python
# Specifické exception handling
try:
    result = risky_operation()
except FileNotFoundError as e:
    logger.error(f"Model file not found: {e}")
    raise
except torch.cuda.OutOfMemoryError:
    logger.warning("CUDA OOM, falling back to CPU")
    return fallback_cpu_operation()
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise RuntimeError(f"Operation failed: {e}") from e
```

#### Logging

```python
import logging

# Setup logging
logger = logging.getLogger(__name__)

# Usage
logger.info("Starting model loading")
logger.warning("Low VRAM detected, enabling optimizations")
logger.error("Failed to load model: %s", model_path)
logger.debug("Processing image with shape: %s", image.shape)
```

### File Organization

```
src/
├── __init__.py
├── models/
│   ├── __init__.py
│   ├── lora_loader.py
│   ├── full_model_loader.py
│   └── base_model.py
├── ui/
│   ├── __init__.py
│   ├── components.py
│   ├── styles.py
│   └── utils.py
├── utils/
│   ├── __init__.py
│   ├── image_processing.py
│   ├── system_info.py
│   └── device_detection.py
└── config/
    ├── __init__.py
    └── settings.py
```

### Configuration Management

```python
# config/settings.py
from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class AppConfig:
    """Application configuration."""
    
    # Paths
    lora_models_path: str = os.getenv('LORA_MODELS_PATH', '/data/loras')
    full_models_path: str = os.getenv('FULL_MODELS_PATH', '/data/models')
    hf_home: str = os.getenv('HF_HOME', '/root/.cache/huggingface')
    
    # Performance
    force_cpu: bool = os.getenv('FORCE_CPU', 'false').lower() == 'true'
    max_memory_gb: float = float(os.getenv('MAX_MEMORY_GB', '8'))
    enable_attention_slicing: bool = os.getenv('ENABLE_ATTENTION_SLICING', 'true').lower() == 'true'
    
    # Model
    base_model: str = os.getenv('BASE_MODEL', 'stabilityai/stable-diffusion-xl-base-1.0')
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_memory_gb <= 0:
            raise ValueError("max_memory_gb must be positive")
        
        if not os.path.exists(self.lora_models_path):
            os.makedirs(self.lora_models_path, exist_ok=True)
```

---

## 🧪 Testing

### Test Structure

```
tests/
├── __init__.py
├── conftest.py
├── unit/
│   ├── test_models.py
│   ├── test_utils.py
│   └── test_config.py
├── integration/
│   ├── test_pipeline.py
│   └── test_ui.py
└── fixtures/
    ├── test_image.png
    └── mock_model.safetensors
```

### Writing Tests

```python
# tests/unit/test_models.py
import pytest
from unittest.mock import Mock, patch
from PIL import Image

from src.models.lora_loader import LoRALoader
from src.config.settings import AppConfig

class TestLoRALoader:
    """Test LoRA model loading functionality."""
    
    @pytest.fixture
    def config(self):
        """Test configuration."""
        return AppConfig(
            lora_models_path="/tmp/test_loras",
            force_cpu=True
        )
    
    @pytest.fixture
    def mock_model_file(self, tmp_path):
        """Create mock model file."""
        model_file = tmp_path / "test_model.safetensors"
        model_file.write_bytes(b"mock model data")
        return str(model_file)
    
    def test_load_lora_success(self, config, mock_model_file):
        """Test successful LoRA loading."""
        loader = LoRALoader(config)
        
        with patch('src.models.lora_loader.load_file') as mock_load:
            mock_load.return_value = {"test_key": "test_value"}
            
            result = loader.load_lora(mock_model_file)
            
            assert result is not None
            mock_load.assert_called_once_with(mock_model_file)
    
    def test_load_lora_file_not_found(self, config):
        """Test LoRA loading with non-existent file."""
        loader = LoRALoader(config)
        
        with pytest.raises(FileNotFoundError):
            loader.load_lora("/non/existent/file.safetensors")
    
    @patch('src.models.lora_loader.torch.cuda.is_available')
    def test_device_detection_no_cuda(self, mock_cuda, config):
        """Test device detection when CUDA is not available."""
        mock_cuda.return_value = False
        loader = LoRALoader(config)
        
        device = loader.get_device()
        assert device == "cpu"
```

### Running Tests

```bash
# Všechny testy
pytest

# S coverage
pytest --cov=src --cov-report=html

# Specifické testy
pytest tests/unit/test_models.py::TestLoRALoader::test_load_lora_success

# Verbose output
pytest -v

# Parallel execution
pytest -n auto
```

### Test Configuration

**`pytest.ini`**
```ini
[tool:pytest]
addopts = 
    --strict-markers
    --strict-config
    --cov=src
    --cov-report=term-missing
    --cov-report=html
    --cov-fail-under=80

testpaths = tests

markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    gpu: marks tests that require GPU
    integration: marks tests as integration tests
    unit: marks tests as unit tests
```

---

## 📖 Dokumentace

### Typy dokumentace

1. **Code documentation** - Docstrings, comments
2. **API documentation** - Automaticky generovaná z docstrings
3. **User guides** - Manuály pro uživatele
4. **Developer docs** - Technická dokumentace

### Docstring Standards

```python
def apply_style_transfer(
    input_image: Image.Image,
    model_path: str,
    strength: float = 0.8,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 20,
    seed: Optional[int] = None
) -> List[Image.Image]:
    """Apply style transfer using LoRA or full model.
    
    This function performs image-to-image style transfer using either
    LoRA adapters or full Stable Diffusion models. The process involves
    loading the specified model, preprocessing the input image, and
    generating stylized variants.
    
    Args:
        input_image: Source image for style transfer. Should be RGB format
            with dimensions between 512x512 and 2048x2048 for optimal results.
        model_path: Absolute path to the model file (.safetensors format).
            For LoRA models, place in data/loras/. For full models, place
            in data/models/.
        strength: Style application strength. Higher values (0.8-1.0) produce
            more dramatic changes, while lower values (0.3-0.6) preserve
            more of the original image structure.
        guidance_scale: Classifier-free guidance scale. Controls how closely
            the model follows the style. Range: 1.0-20.0, recommended: 7.5.
        num_inference_steps: Number of denoising steps. More steps generally
            improve quality but increase generation time. Range: 10-50.
        seed: Random seed for reproducible results. If None, uses random seed.
    
    Returns:
        List of generated PIL Images. Length depends on batch_count parameter
        in the UI. Each image is in RGB format with the same dimensions as
        the input image (unless upscaling is enabled).
    
    Raises:
        FileNotFoundError: If the model file doesn't exist at the specified path.
        RuntimeError: If CUDA runs out of memory or model loading fails.
        ValueError: If input parameters are outside valid ranges.
        TypeError: If input_image is not a PIL Image object.
    
    Example:
        >>> from PIL import Image
        >>> img = Image.open("portrait.jpg")
        >>> results = apply_style_transfer(
        ...     input_image=img,
        ...     model_path="/data/loras/tuymans_style.safetensors",
        ...     strength=0.8,
        ...     guidance_scale=7.5,
        ...     num_inference_steps=25
        ... )
        >>> len(results)
        1
        >>> results[0].save("stylized_portrait.png")
    
    Note:
        - GPU with 8GB+ VRAM recommended for optimal performance
        - CPU fallback is available but significantly slower
        - Model caching is enabled to speed up subsequent generations
        - Memory is automatically cleaned up after generation
    
    See Also:
        load_base_model: For loading and caching base models
        load_full_model: For loading and caching full models
        get_optimal_device: For automatic device detection
    """
    pass
```

### README Updates

Při přidávání nových funkcí aktualizujte:

1. **Feature list** v README.md
2. **Usage examples**
3. **API documentation** odkazy
4. **Changelog** sekci

---

## 🔄 Pull Request Proces

### Před vytvořením PR

1. **Aktualizujte váš fork:**
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   git push origin main
   ```

2. **Vytvořte feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   # nebo
   git checkout -b fix/bug-description
   ```

3. **Implementujte změny:**
   - Dodržujte coding standards
   - Přidejte testy
   - Aktualizujte dokumentaci

4. **Testování:**
   ```bash
   # Linting
   black .
   isort .
   flake8 .
   mypy .
   
   # Tests
   pytest
   
   # Pre-commit hooks
   pre-commit run --all-files
   ```

### Commit Messages

Používáme **Conventional Commits** formát:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Typy:**
- `feat`: Nová funkce
- `fix`: Oprava chyby
- `docs`: Dokumentace
- `style`: Formátování (ne CSS)
- `refactor`: Refaktoring
- `test`: Testy
- `chore`: Maintenance

**Příklady:**
```bash
# Nová funkce
git commit -m "feat(models): add support for SDXL LoRA models"

# Oprava chyby
git commit -m "fix(ui): resolve memory leak in image display"

# Dokumentace
git commit -m "docs(api): add examples for apply_style function"

# Breaking change
git commit -m "feat(api): change model loading interface

BREAKING CHANGE: model_path parameter now requires absolute path"
```

### PR Template

```markdown
## 📝 Description

Brief description of changes and motivation.

## 🔗 Related Issues

Fixes #123
Closes #456

## 🧪 Testing

- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Performance impact assessed

## 📋 Checklist

- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added for new functionality
- [ ] All tests pass
- [ ] No breaking changes (or documented)

## 🖼️ Screenshots

<!-- If applicable, add screenshots -->

## 📚 Additional Notes

<!-- Any additional information -->
```

### Review Process

1. **Automated checks** musí projít:
   - CI/CD pipeline
   - Code quality checks
   - Test coverage

2. **Code review** od maintainerů:
   - Funkčnost
   - Code quality
   - Performance
   - Security

3. **Approval a merge:**
   - Minimálně 1 approval
   - Všechny checks zelené
   - Conflicts vyřešené

---

## 🐛 Bug Reporting

### Před hlášením

1. **Zkontrolujte existující issues**
2. **Aktualizujte na nejnovější verzi**
3. **Reprodukujte problém**
4. **Shromážděte informace**

### Bug Report Template

```markdown
## 🐛 Bug Report

**Popis:**
Jasný a stručný popis problému.

**Reprodukce:**
Kroky k reprodukci:
1. Jděte na '...'
2. Klikněte na '....'
3. Scrollujte dolů na '....'
4. Vidíte chybu

**Očekávané chování:**
Jasný popis toho, co jste očekávali.

**Screenshots:**
Pokud je to možné, přidejte screenshots.

**Prostředí:**
 - OS: [e.g. Ubuntu 22.04]
 - Python: [e.g. 3.11.5]
 - GPU: [e.g. RTX 4090]
 - Docker: [Yes/No]
 - Verze: [e.g. v2.0.0]

**Logy:**
```
Vložte relevantní logy zde
```

**Dodatečný kontext:**
Jakékoliv další informace o problému.
```

---

## 💡 Feature Requests

### Feature Request Template

```markdown
## 💡 Feature Request

**Je váš feature request spojen s problémem?**
Jasný popis problému. Ex. Jsem vždy frustrovaný když [...]

**Popište řešení, které byste chtěli:**
Jasný a stručný popis toho, co chcete, aby se stalo.

**Popište alternativy, které jste zvážili:**
Jasný popis alternativních řešení nebo funkcí, které jste zvážili.

**Dodatečný kontext:**
Přidejte jakýkoliv další kontext nebo screenshots o feature requestu zde.

**Implementace:**
- [ ] Jsem ochoten implementovat tuto funkci
- [ ] Potřebuji pomoc s implementací
- [ ] Pouze navrhuji, neimplementuji
```

### Diskuze nových funkcí

Velké změny diskutujte v **GitHub Discussions** před implementací:

1. **Návrh architektury**
2. **API design**
3. **Performance implications**
4. **Breaking changes**

---

## 👥 Community

### Code of Conduct

Dodržujeme [Contributor Covenant](https://www.contributor-covenant.org/):

- **Buďte respektující** - Respektujte různé názory a zkušenosti
- **Buďte konstruktivní** - Poskytujte konstruktivní feedback
- **Buďte trpěliví** - Pamatujte, že všichni jsou dobrovolníci
- **Buďte inkluzivní** - Vítáme přispěvatele všech úrovní

### Communication Channels

- **GitHub Issues** - Bug reports, feature requests
- **GitHub Discussions** - Obecné diskuze, Q&A
- **Discord** - Real-time chat (link v README)
- **Email** - Soukromé dotazy

### Recognition

Přispěvatelé jsou uznáváni:

- **Contributors list** v README
- **Release notes** mentions
- **Special badges** pro významné příspěvky
- **Maintainer status** pro dlouhodobé přispěvatele

---

## 🎉 Děkujeme!

Děkujeme za váš zájem přispět do projektu LoRA Tuymans Style Transfer! Každý příspěvek, ať už malý nebo velký, je vítán a oceňován.

**Happy coding! 🚀**

---

**📧 Kontakt:** [your-email@example.com](mailto:your-email@example.com)  
**🐛 Issues:** [GitHub Issues](https://github.com/your-username/lora_tuymans/issues)  
**💬 Diskuze:** [GitHub Discussions](https://github.com/your-username/lora_tuymans/discussions)