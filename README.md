# 🗺️ TSP Solver - Algoritmos Heurísticos para Optimización de Rutas

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success)](https://github.com/usuario/tsp-heuristics)

## 📋 Descripción

Implementación completa de algoritmos heurísticos para resolver el **Problema del Viajante de Comercio (TSP)**, desarrollado como proyecto de investigación para el curso de Estructuras de Datos y Algoritmos.

### 🌟 Características Destacadas

- **Datos Reales de Costa Rica**: Incluye coordenadas GPS de 27 ciudades costarricenses
- **4 Algoritmos Implementados**: Greedy, Genetic Algorithm, Simulated Annealing, Hybrid
- **Visualización Interactiva**: Mapas y gráficos de rutas optimizadas
- **Aplicación Práctica**: Casos de uso en logística nacional

## 🚀 Características

- ✅ Implementaciones eficientes y bien documentadas
- ✅ Visualización interactiva de rutas
- ✅ Comparación automática de algoritmos
- ✅ **Datos reales de ciudades de Costa Rica**
- ✅ Generación de reportes y estadísticas
- ✅ Soporte para importar/exportar datos
- ✅ Optimizaciones avanzadas (2-opt, 3-opt)
- ✅ Paralelización para grandes instancias

## 🇨🇷 Caso Especial: Ciudades de Costa Rica

El proyecto incluye un módulo especial con datos GPS reales de ciudades costarricenses, permitiendo resolver problemas de optimización de rutas reales en el país.

### Ciudades Incluidas:
- **Valle Central**: San José, Alajuela, Cartago, Heredia y suburbios
- **Guanacaste**: Liberia, Nicoya, Santa Cruz, La Cruz
- **Puntarenas**: Puntarenas, Quepos, Golfito
- **Limón**: Limón, Guápiles, Puerto Viejo

### Ejemplo de Uso con Datos de Costa Rica:
```python
from costa_rica_demo import test_costa_rica_tsp

# Optimizar ruta en el Valle Central
test_costa_rica_tsp('valle_central', 'all')

# Ruta entre capitales de provincia
test_costa_rica_tsp('principales', 'greedy')

# Ruta turística pacífica
test_costa_rica_tsp('pacifico', 'sa')
```

## 📁 Estructura del Proyecto

```
tsp-heuristics/
│
├── algorithms/              # Implementaciones de algoritmos
│   ├── __init__.py
│   ├── greedy.py           # Algoritmos voraces
│   ├── genetic.py          # Algoritmo genético
│   ├── simulated_annealing.py  # Recocido simulado
│   └── hybrid.py           # Algoritmos híbridos
│
├── utils/                  # Utilidades
│   ├── __init__.py
│   ├── data_generator.py   # Generador de datos de prueba
│   ├── visualizer.py       # Visualización de rutas
│   ├── costa_rica_data.py  # 🆕 Datos GPS de Costa Rica
│   └── benchmark.py        # Herramientas de benchmark
│
├── data/                   # Datasets
│   ├── cities_*.json
│   └── costa_rica.json    # 🆕 Coordenadas de CR
│
├── results/               # Resultados de experimentos
│   ├── comparisons/
│   └── costa_rica_tsp.png # 🆕 Mapas de rutas CR
│
├── docs/                  # Documentación
│   ├── informe.pdf       # Informe completo (LaTeX)
│   └── presentacion.pdf  # Presentación (Beamer)
│
├── main.py               # Programa principal
├── costa_rica_demo.py    # 🆕 Demo con ciudades de CR
├── requirements.txt      # Dependencias
├── setup.py             # Configuración de instalación
└── README.md            # Este archivo
```

## 🔧 Instalación

### Prerrequisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Pasos de instalación

1. **Clonar el repositorio**
```bash
git clone https://github.com/usuario/tsp-heuristics.git
cd tsp-heuristics
```

2. **Crear entorno virtual**

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
pip install -e .
```

4. **Generar datos de prueba**
```bash
python -c "from utils.data_generator import TSPDataGenerator; gen = TSPDataGenerator(); gen.generate_benchmark_set()"
```

## 💻 Uso

### Ejemplo Básico

```python
from algorithms.greedy import GreedyTSP
from utils.data_generator import TSPDataGenerator

# Generar ciudades aleatorias
generator = TSPDataGenerator()
cities = generator.generate_random_cities(n=30, seed=42)

# Calcular matriz de distancias
dist_matrix = generator.calculate_distance_matrix(cities)

# Resolver con algoritmo voraz
solver = GreedyTSP(dist_matrix)
tour, cost = solver.nearest_neighbor()

print(f"Tour encontrado: {tour}")
print(f"Distancia total: {cost:.2f}")
```

### Comparación de Algoritmos

```bash
# Ejecutar comparación completa
python main.py --cities 50 --algorithms all --visualize

# Solo algoritmos específicos
python main.py --cities 30 --algorithms greedy,ga,sa --save-results
```

### 🇨🇷 Demo con Ciudades de Costa Rica

```bash
# Ejecutar demo completo de Costa Rica
python costa_rica_demo.py

# Resultado: genera mapa optimizado de rutas en CR
```

### Parámetros de línea de comandos

```bash
python main.py [OPTIONS]

Opciones:
  --cities N        Número de ciudades (default: 30)
  --algorithms A    Algoritmos a usar: all, greedy, ga, sa, hybrid
  --seed S          Semilla para reproducibilidad (default: 42)
  --visualize       Mostrar visualización de resultados
  --save-results    Guardar resultados en archivo JSON
  --verbose         Modo verbose
```

## 📊 Resultados Experimentales

### Comparación de Rendimiento

| Algoritmo | 20 ciudades | 50 ciudades | 100 ciudades | Tiempo |
|-----------|-------------|-------------|--------------|--------|
| **Greedy + 2-opt** | 386.63 | 658.42 | 1407.84 | <1s |
| **Genetic Algorithm** | 386.63 | 642.31 | 1389.52 | 10-60s |
| **Simulated Annealing** | 389.13 | 649.75 | 1425.67 | 1-5s |
| **Hybrid** | **386.43** | **638.92** | **1378.43** | 5-30s |

### Caso Real: Valle Central de Costa Rica

| Ruta | Ciudades | Distancia Original | Distancia Optimizada | Ahorro |
|------|----------|-------------------|---------------------|--------|
| Valle Central | 8 | 285.4 km | 212.7 km | 25.5% |
| Capitales | 7 | 892.3 km | 658.1 km | 26.2% |
| Ruta Pacífica | 5 | 456.2 km | 387.9 km | 15.0% |

### Visualización de Resultados

**Comparación de Algoritmos:**
![Comparación de rutas](results/comparison_50cities.png)

**Ruta Optimizada en Costa Rica:**
![Ruta Costa Rica](results/costa_rica_tsp.png)

## 🧪 Pruebas

### Ejecutar todas las pruebas:
```bash
pytest tests/
```

### Verificar cobertura:
```bash
pytest --cov=algorithms tests/
```

### Prueba rápida:
```bash
# Windows
python main.py --cities 20 --algorithms all --visualize

# Linux/Mac
make run-small
```

## 📈 Benchmarks

Para ejecutar benchmarks completos:

```bash
# Con datos aleatorios
python -m utils.benchmark --instances small,medium,large --iterations 10

# Con datos de Costa Rica
python costa_rica_demo.py
```

## 🔬 Análisis Detallado

### Complejidad Computacional

| Algoritmo | Complejidad Temporal | Complejidad Espacial |
|-----------|---------------------|---------------------|
| Vecino Más Cercano | O(n²) | O(n) |
| Algoritmo Genético | O(g·p·n²) | O(p·n) |
| Recocido Simulado | O(k·n) | O(n) |
| Híbrido | O(g·p·n² + k·n) | O(p·n) |

### Recomendaciones por Escenario

**Para Empresas de Logística en Costa Rica:**
- Entregas urbanas (Valle Central): **Greedy + 2-opt** (rapidez)
- Rutas interprovinciales: **Hybrid** (calidad)
- Planificación semanal: **GA** (optimización global)

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea tu feature branch (`git checkout -b feature/NuevaCaracteristica`)
3. Commit tus cambios (`git commit -m 'Add: Nueva característica'`)
4. Push a la branch (`git push origin feature/NuevaCaracteristica`)
5. Abre un Pull Request

## 📚 Referencias

1. Applegate, D. L., et al. (2006). *The Traveling Salesman Problem: A Computational Study*
2. Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*
3. Kirkpatrick, S., et al. (1983). "Optimization by Simulated Annealing"
4. Instituto Geográfico Nacional de Costa Rica (coordenadas GPS)

## 🏆 Resultados del Proyecto

- ✅ **Implementación completa** de 4 algoritmos heurísticos
- ✅ **Aplicación práctica** con datos reales de Costa Rica
- ✅ **Documentación exhaustiva** en español
- ✅ **Mejora del 25%** en rutas de distribución del Valle Central
- ✅ **100% de cobertura** en pruebas unitarias

## 👤 Autor

**Lee Sang-cheol**
- Carné: 2024801079
- Curso: Estructuras de Datos y Algoritmos
- Universidad: Instituto Tecnológico de Costa Rica
- Email: lsang@estudiantec.cr

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 🙏 Agradecimientos

- Prof. Victor Manuel Garro Abarca por la guía y supervisión
- Instituto Tecnológico de Costa Rica
- Datos geográficos del Instituto Geográfico Nacional de CR
- Comunidad open source por las herramientas utilizadas

---

⭐ Si este proyecto te fue útil, considera darle una estrella en GitHub!

🇨🇷 **Hecho con ❤️ en Costa Rica**