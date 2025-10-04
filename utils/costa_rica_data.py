"""
Datos reales de ciudades de Costa Rica para TSP
Coordenadas en formato (latitud, longitud)
"""

# Ciudades principales de Costa Rica con coordenadas GPS
COSTA_RICA_CITIES = [
    # Provincia de San José
    (9.9281, -84.0907, "San José (Capital)"),
    (9.8602, -83.9211, "San Isidro de El General"),
    (9.9739, -84.0163, "Escazú"),
    (9.9369, -84.0413, "Santa Ana"),
    
    # Provincia de Alajuela
    (10.0160, -84.2163, "Alajuela (Centro)"),
    (10.4763, -84.6452, "San Carlos"),
    (10.0875, -84.4703, "Grecia"),
    (10.0729, -84.3116, "Atenas"),
    
    # Provincia de Cartago
    (9.8639, -83.9193, "Cartago (Centro)"),
    (9.8492, -83.8652, "Paraíso"),
    (9.7949, -83.7888, "Turrialba"),
    
    # Provincia de Heredia
    (9.9981, -84.1197, "Heredia (Centro)"),
    (10.0950, -84.0963, "Barva"),
    (10.0117, -84.0969, "Santo Domingo"),
    
    # Provincia de Guanacaste
    (10.6345, -85.4406, "Liberia"),
    (10.5425, -85.6978, "La Cruz"),
    (10.4472, -85.2517, "Nicoya"),
    (10.1414, -85.4522, "Santa Cruz"),
    
    # Provincia de Puntarenas
    (9.9762, -84.8384, "Puntarenas (Centro)"),
    (8.4241, -83.3285, "San Vito"),
    (9.3673, -84.3820, "Quepos"),
    (8.7103, -83.0511, "Golfito"),
    
    # Provincia de Limón
    (9.9907, -83.0309, "Limón (Centro)"),
    (10.1940, -83.3743, "Guápiles"),
    (9.6432, -82.6368, "Sixaola"),
    (9.7339, -82.9598, "Puerto Viejo"),
]

def get_costa_rica_cities():
    """
    Retorna las ciudades de Costa Rica en formato para TSP
    
    Returns:
        cities: Lista de tuplas (lat, lon)
        names: Lista de nombres de ciudades
    """
    cities = [(lat, lon) for lat, lon, _ in COSTA_RICA_CITIES]
    names = [name for _, _, name in COSTA_RICA_CITIES]
    return cities, names

def create_costa_rica_distance_matrix():
    """
    Crea matriz de distancias usando la fórmula de Haversine
    para distancias geográficas reales
    """
    import math
    
    cities, _ = get_costa_rica_cities()
    n = len(cities)
    dist_matrix = [[0] * n for _ in range(n)]
    
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calcula distancia en km entre dos puntos GPS"""
        R = 6371  # Radio de la Tierra en km
        
        # Convertir a radianes
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Diferencias
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        # Fórmula de Haversine
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    # Calcular todas las distancias
    for i in range(n):
        for j in range(i+1, n):
            dist = haversine_distance(
                cities[i][0], cities[i][1],
                cities[j][0], cities[j][1]
            )
            dist_matrix[i][j] = dist_matrix[j][i] = dist
    
    return dist_matrix, cities

# Subconjuntos útiles para pruebas
COSTA_RICA_SUBSETS = {
    'valle_central': [0, 2, 3, 4, 8, 11, 12, 13],  # San José y alrededores
    'principales': [0, 4, 8, 11, 14, 19, 23],  # Capitales de provincia
    'pacifico': [0, 19, 20, 21, 22],  # Ruta pacífica
    'caribe': [23, 24, 25, 26],  # Ruta caribeña
    'completo': list(range(len(COSTA_RICA_CITIES)))  # Todas las ciudades
}