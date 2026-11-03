# 🗺️ TSP Solver - Algoritmos Heurísticos para Optimización de Rutas

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE.md)

> **English summary.** Solo course project for *Data Structures and Algorithms* at Instituto Tecnológico de Costa Rica (Oct 2025). Python implementations of four TSP heuristics: nearest-neighbour greedy with 2-opt/3-opt, a genetic algorithm, simulated annealing, and a hybrid (greedy → GA → SA → local search).
>
> - **Benchmark** (`results/benchmarks/report_20251004_142803.txt`): random instances of 10, 15, 20, 25 and 30 cities, 5 iterations each. Average tour cost / time: hybrid 385.28 / 5.53 s, greedy 393.27 / 0.06 s, GA 400.26 / 4.75 s, SA 415.22 / 0.06 s.
> - **Costa Rica demo** (`costa_rica_demo.py`): GPS coordinates of 26 cities with haversine distances; runs greedy, SA and (for ≤ 20 cities) GA on city subsets and plots the routes.
> - **Tests:** `pytest tests/` runs 15 unit tests for the greedy solver (they pass; statement coverage of `algorithms/` is 25 %). GA, SA and hybrid have no unit tests.
> - The rest of this README is the original Spanish documentation, corrected in Nov 2026 where it did not match the code (see *Notas de corrección* at the end).

## 🎥 Video Demostración

[![Ver Video Demo](https://img.shields.io/badge/▶️-Ver_Video_Demo-red?style=for-the-badge&logo=youtube)](https://youtu.be/TwWuK-tHaSM)

**→ [Ver demostración completa del proyecto en YouTube](https://youtu.be/TwWuK-tHaSM)**


## 📋 Descripción

Implementación completa de algoritmos heurísticos para resolver el **Problema del Viajante de Comercio (TSP)**, desarrollado como proyecto de investigación para el curso de Estructuras de Datos y Algoritmos.

### 🌟 Características Destacadas

- **Datos Reales de Costa Rica**: Incluye coordenadas GPS de 26 ciudades costarricenses
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
- ✅ Ejecución paralela opcional en el algoritmo híbrido (`ThreadPoolExecutor`)

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
Proyecto2-tsp-heuristics/
├── algorithms/          # greedy.py, genetic.py, simulated_annealing.py, hybrid.py
├── utils/               # data_generator.py, visualizer.py, costa_rica_data.py, benchmark.py
├── tests/test_greedy.py # 15 pruebas unitarias (solo Greedy)
├── results/benchmarks/  # CSV/JSON/informes de benchmark (2025-09-30, 2025-10-04)
├── docs/                # informe, presentación, hoja de resumen, figuras
├── main.py              # demo: 20, 50 y 100 ciudades aleatorias
├── costa_rica_demo.py   # demo con ciudades de Costa Rica
├── benchmark.py
├── Makefile.txt
├── requirements.txt
└── setup.py
```

## 🔧 Instalación

### Prerrequisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Pasos de instalación

1. **Clonar el repositorio**
```bash
git clone https://github.com/DIEGO-LEE-24/Proyecto2-tsp-heuristics.git
cd Proyecto2-tsp-heuristics
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
# main.py no recibe argumentos: compara los 4 algoritmos con 20, 50 y 100
# ciudades aleatorias (seed=42) y pide Enter entre cada tamaño
python main.py
```

### 🇨🇷 Demo con Ciudades de Costa Rica

```bash
# Ejecuta los subconjuntos valle_central, principales y pacifico
# y guarda el mapa en results/costa_rica_tsp.png
python costa_rica_demo.py
```

## 📊 Resultados Experimentales

### Comparación de Rendimiento

Fuente: `results/benchmarks/report_20251004_142803.txt` — instancias aleatorias de 10, 15, 20, 25 y 30 ciudades, 5 iteraciones cada una.

| Algoritmo | Costo promedio | Tiempo promedio |
|-----------|----------------|-----------------|
| **Hybrid** | **385.28** | 5.531 s |
| Greedy | 393.27 | 0.062 s |
| Genetic Algorithm | 400.26 | 4.754 s |
| Simulated Annealing | 415.22 | 0.064 s |

## 🧪 Pruebas

### Ejecutar las pruebas:
```bash
pytest tests/
```

Hay 15 pruebas unitarias, todas para `GreedyTSP`. Con `pytest --cov=algorithms tests/` la cobertura medida es 25 % (greedy.py 91 %; genetic, simulated_annealing e hybrid casi sin cubrir).

## 📈 Benchmarks

Para ejecutar benchmarks completos:

```bash
# Con datos aleatorios
python -m utils.benchmark --instances small --iterations 10 --plot

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
- ✅ Benchmark reproducible con resultados guardados en `results/benchmarks/`

## 👤 Autor

**Lee Sang-cheol**
- Curso: Estructuras de Datos y Algoritmos
- Universidad: Instituto Tecnológico de Costa Rica

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE.md) para más detalles.

## 🙏 Agradecimientos

- Prof. Victor Manuel Garro Abarca por la guía y supervisión
- Instituto Tecnológico de Costa Rica
- Datos geográficos del Instituto Geográfico Nacional de CR
- Comunidad open source por las herramientas utilizadas

---

⭐ Si este proyecto te fue útil, considera darle una estrella en GitHub!

🇨🇷 **Hecho con ❤️ en Costa Rica**

## Notas de corrección (nov. 2026)

- El número de ciudades de Costa Rica es 26 (`COSTA_RICA_CITIES`); antes decía 27.
- Se eliminó la tabla de 50/100 ciudades y la tabla de ahorro del Valle Central (25.5 %): venían del informe del curso y la ruta "original" usada como referencia no está en este repositorio, así que no se pueden reproducir. La tabla actual sale de los resultados guardados.
- Se eliminó la afirmación de "100 % de cobertura"; la cobertura medida es 25 %.
- `main.py` no tiene opciones de línea de comandos; se eliminaron los ejemplos con `--cities` y `make run-small` (el Makefile está guardado como `Makefile.txt`).
- Problema conocido: el subconjunto `caribe` usa el índice 26, que no existe (hay 26 ciudades, índices 0–25), así que `test_costa_rica_tsp('caribe')` falla. No se usa en la demo.
