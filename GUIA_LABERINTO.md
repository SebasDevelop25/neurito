# Neurito 3.0: aprende a encontrar comida en un laberinto

Neurito busca comida dentro de tres laberintos. Su cerebro debe elegir entre arriba, abajo, izquierda y derecha. La red se construye con NumPy, y Pygame dibuja el escenario, la mascota y una gráfica de la pérdida.

## Cómo aprende

1. BFS calcula rutas cortas y genera ejemplos etiquetados: mapa, posición, meta y movimiento recomendado.
2. Los ejemplos se dividen en entrenamiento (80 %) y prueba (20 %). Los pares posición-meta de prueba no aparecen en entrenamiento.
3. Cada entrada incluye un vector one-hot de tres mapas y dos vectores one-hot de 81 casillas: 165 características.
4. La arquitectura es 165 → 96 → 4. Usa ReLU en la capa oculta y Softmax en las cuatro salidas.
5. El entrenamiento minimiza entropía cruzada multiclase mediante retropropagación y descenso de gradiente con mini-lotes.
6. Durante el juego, el modelo decide sin consultar BFS. BFS es el experto que proporcionó las etiquetas.

Esto es aprendizaje supervisado por imitación, NO aprendizaje por refuerzo.

## Dos modos

- Red pura: toma la salida de mayor probabilidad. Puede chocar con paredes o quedarse en ciclos.
- Asistido: descarta movimientos contra paredes y penaliza casillas visitadas. Son reglas adicionales, no conocimientos aprendidos por la red.

La precisión de clasificación no equivale al porcentaje de laberintos completados.

## Estructura

- laberinto_mundo.py: mapas, movimiento, BFS y navegación asistida.
- laberinto_red.py: red neuronal y entrenamiento con NumPy.
- laberinto.py: juego Pygame, sprites y pérdida.
- test_laberinto.py: pruebas automáticas.
- GUIA_LABERINTO.md: esta guía.

Reutiliza assets/neutral_1.png, neutral_2.png y neutral_3.png. No modifica las versiones anteriores.

## Ejecutar

    uv sync
    uv run laberinto.py

## Controles

| Tecla | Acción |
|---|---|
| 1 | Entrenar una época |
| 2 | Entrenar cien épocas |
| 3 | Entrenamiento continuo |
| A | Alternar red pura y navegación asistida |
| P | Alternar navegación automática y paso a paso |
| Espacio | Avanzar un paso en modo manual |
| N | Cambiar mapa |
| G | Nuevas posiciones y meta |
| C | Reiniciar contadores |
| R | Reiniciar pesos y entrenamiento |
| Esc | Salir |

## Probar automáticamente

    uv run python -m unittest -v test_laberinto.py

## Experimentos

### Cómo se midió

Todo se ejecuta sin Pygame con [experimentos_laberinto.py](experimentos_laberinto.py):

    uv run python experimentos_laberinto.py        # los 7 experimentos (tarda varios minutos)
    uv run python experimentos_laberinto.py 2 4    # solo algunos

La salida completa quedó en [resultados_experimentos.txt](resultados_experimentos.txt). Cada celda es el **promedio de 5 semillas** de inicialización de pesos (0 a 4). La partición entrenamiento/prueba es siempre la misma (semilla 42).

| Columna | Significado |
|---|---|
| pérdida | entropía cruzada de la última época (entrenamiento) |
| acc.tr / acc.te | porcentaje de veces que el argmax coincide con el movimiento del profesor, en entrenamiento / prueba |
| éxito | fracción de partidas que llegan a la comida en 60 intentos, sobre hasta 400 pares de prueba no entrenados |
| choq | choques contra paredes por partida |
| extra | pasos dados ÷ pasos óptimos de BFS, solo en partidas exitosas (1.00 = ruta óptima) |

### 1. Neurito sin entrenar: ¿por qué se atasca?

| Modo | Éxito | Choques por partida |
|---|---|---|
| Red pura | 3.4 % | 51.4 |

Con pesos aleatorios la red tiene una confianza media de solo 0.30 (con 4 acciones el azar puro sería 0.25) y **elige casi siempre la misma dirección**: de las pares de prueba eligió ARRIBA 470 veces, ABAJO 23, IZQUIERDA 125 y DERECHA 98, mientras que el experto reparte las respuestas correctas de forma casi uniforme (184, 188, 175, 169). La exactitud (24.5 %) es la del azar. Como la decisión depende solo de la entrada y no tiene memoria, en cuanto Neurito se topa con una pared repite la misma orden indefinidamente: se queda quieto chocando. No se "atasca" por mala suerte; la función que calcula es casi constante.

### 2. Entrenar 100, 200 y 400 épocas

| Épocas | Pérdida | acc.tr | acc.te | Éxito (pura) | Choques | Extra |
|---|---|---|---|---|---|---|
| 0 | n/a | 0.264 | 0.245 | 0.034 | 51.4 | n/a |
| 100 | 0.1354 | 0.963 | 0.867 | 0.620 | 1.4 | 1.01 |
| 200 | 0.0389 | 0.999 | 0.898 | 0.807 | 0.7 | 1.01 |
| 400 | 0.0110 | 1.000 | 0.908 | 0.835 | 0.6 | 1.01 |

- La pérdida cae un orden de magnitud cada vez que se duplican las épocas, pero la exactitud de prueba se estanca en torno al 91 %.
- Con 200 épocas la exactitud de entrenamiento ya es 99.9 %: de ahí en adelante la red **memoriza** el conjunto de entrenamiento y la brecha con prueba (≈ 9 puntos) deja de cerrarse. Hay algo de sobreajuste, porque se usan 165 entradas one-hot para memorizar casillas concretas, no una noción de geometría.
- La exactitud de prueba (90.8 %) y el éxito jugando (83.5 %) **no son lo mismo**: en un trayecto de 15 pasos basta un movimiento erróneo para entrar en un ciclo. Cuando la red sí llega, lo hace casi por la ruta óptima (extra 1.01).

### 3. Red pura frente a navegación asistida

| Configuración | Éxito pura | Éxito asistida | Extra pura | Extra asistida | Choques pura / asistida |
|---|---|---|---|---|---|
| 0 épocas | 3.4 % | **98.2 %** | n/a | 4.42 | 51.4 / 0.0 |
| 100 épocas | 62.0 % | 100 % | 1.01 | 1.35 | 1.4 / 0.0 |
| 400 épocas | 83.5 % | 100 % | 1.01 | 1.22 | 0.6 / 0.0 |

Este es el resultado más importante de la guía: **la asistencia por sí sola resuelve casi todo**. Sin entrenar, el modo asistido logra 98.2 % porque las reglas (no chocar con paredes y penalizar casillas ya visitadas) equivalen a una exploración con memoria, parecida a un DFS. La diferencia es el costo: sin entrenar da 4.42 veces la ruta óptima; tras 400 épocas, 1.22.

Reparto de responsabilidades:
- **Reglas** (no aprendidas): 0 choques, y evitar ciclos, y por tanto el 100 % de éxito.
- **Modelo**: la *eficiencia*. Pasar de 4.42× a 1.22× el camino óptimo, y de 3 % a 83 % de éxito sin ayuda, es mérito del entrenamiento.

El 100 % del modo asistido, por tanto, **no debe presentarse como prueba de que la red aprendió**. Las métricas válidas sobre la red son las de la columna pura y el extra del modo asistido.

### 4. Neuronas ocultas: 16, 32, 96 y 128 (400 épocas)

| Ocultas | Pérdida | acc.tr | acc.te | Éxito (pura) | Extra asistida |
|---|---|---|---|---|---|
| 16 | 0.0357 | 0.991 | 0.885 | 0.785 | 1.27 |
| 32 | 0.0160 | 1.000 | 0.906 | 0.839 | 1.20 |
| 96 | 0.0110 | 1.000 | 0.908 | 0.835 | 1.22 |
| 128 | 0.0101 | 1.000 | 0.908 | 0.833 | 1.18 |

Con 16 neuronas hay un déficit de capacidad: no llega a memorizar el entrenamiento (99.1 %) y es peor en prueba (88.5 %) y jugando (78.5 %). A partir de 32 la mejora desaparece: 96 y 128 neuronas dan lo mismo que 32 dentro del ruido entre semillas. Para este problema 32 neuronas bastan, y las 96 originales son un margen que no cuesta nada pero tampoco aporta.

### 5. Tasa de aprendizaje (400 épocas)

| Tasa | Pérdida | acc.te | Éxito (pura) | Choques |
|---|---|---|---|---|
| 0.01 | 0.6918 | 0.726 | 0.191 | 20.4 |
| 0.1 | 0.1540 | 0.863 | 0.621 | 1.5 |
| **0.45** | 0.0110 | 0.908 | 0.835 | 0.6 |
| 1.0 | 0.0029 | 0.917 | 0.849 | 0.7 |
| 2.0 | 0.0011 | 0.926 | 0.869 | 0.8 |
| 5.0 | 1.1846 | 0.397 | 0.058 | 44.4 |

- **0.45 converge** (pérdida 0.011, sin oscilaciones).
- Valores bajos (0.01 y 0.1) convergen, pero demasiado lento para 400 épocas: con 0.01 la pérdida sigue en 0.69.
- Subir a 1.0 y 2.0 mejora ligeramente, así que 0.45 es conservadora, no la óptima.
- Con 5.0 el entrenamiento **diverge**: la pérdida termina en 1.18 con exactitud 41 %. Las actualizaciones son tan grandes que saltan sobre el mínimo, y la red queda peor que la de 100 épocas.
- Solo se midió en 400 épocas y 5 semillas; el valor de divergencia exacto estaría entre 2.0 y 5.0 y no se afinó.

### 6. Cambio de mapa

| Mapa | acc.te | Éxito pura | Éxito asistida |
|---|---|---|---|
| 0 | 0.910 | 0.832 | 1.000 |
| 1 | 0.915 | 0.818 | 1.000 |
| 2 | 0.901 | 0.849 | 1.000 |

El rendimiento es prácticamente igual en los tres mapas (diferencias de uno o dos puntos, dentro del ruido). Esto es lo esperado, pero conviene no sobreinterpretarlo: la red recibe el identificador del mapa y entrena con los tres, así que **solo** se mide generalización a combinaciones inicio-meta no vistas, no a laberintos nuevos. Para probar lo segundo habría que codificar las paredes como entrada y dejar un mapa completo fuera del entrenamiento (ver Alcance y limitaciones).

### 7. Profesor: DFS, BFS y A*

Se cambió el algoritmo que genera las etiquetas (se entrena con el primer paso de la ruta que da cada algoritmo; la red y el resto del código no cambian):

| Profesor | Ruta del profesor ÷ óptima | acc.te | Éxito (pura) | Éxito (asistida) | Extra asistida |
|---|---|---|---|---|---|
| DFS | 2.10 | 0.946 | **0.092** | **0.692** | 4.24 |
| BFS | 1.00 | 0.908 | 0.835 | 1.000 | 1.22 |
| A* | 1.00 | 0.914 | 0.852 | 1.000 | 1.11 |

- **DFS empeora mucho a Neurito**, y es el resultado más llamativo: tiene la *mejor* exactitud de prueba (94.6 %) y el *peor* éxito jugando (9.2 % pura). Explicación probable (no se verificó con un experimento aparte): Primero, DFS no da rutas cortas (en promedio el doble de la óptima). Segundo, y más grave, las etiquetas de DFS **no son consistentes entre casillas vecinas**: la ruta de DFS desde A pasa por B, pero la ruta desde B puede ser otra distinta y alejarse de la meta. Eso es fácil de imitar casilla por casilla (alta exactitud), pero al encadenar los movimientos produce ciclos. Aquí la exactitud de clasificación engaña.
- Hasta la asistencia lo sufre: 69.2 % y 4.24× el camino óptimo.
- **BFS y A* son prácticamente equivalentes** porque ambos dan rutas óptimas; sus etiquetas difieren solo en el desempate entre caminos igual de cortos. A* queda ligeramente mejor (85.2 % frente a 83.5 % puro, y extra 1.11 frente a 1.22). Con 5 semillas esa diferencia es pequeña y no es concluyente.
- En un laberinto de 81 casillas A* no ofrece ventaja de costo respecto a BFS (esa ventaja aparece en mapas grandes). Aquí importa solo que las etiquetas sean consistentes y óptimas.

### Conclusiones

1. El entrenamiento convierte una red que choca sin parar (3 %) en una que llega a la comida el 83 % de las veces y casi por el camino más corto.
2. La exactitud de prueba (≈ 91 %) sobrestima el desempeño jugando, y la exactitud de entrenamiento no sirve como medida: llega a 100 % por memorización.
3. El modo asistido hace casi todo el trabajo de completar el laberinto. Hay que reportar siempre el modo puro.
4. Con este problema 32 neuronas son suficientes, y la tasa 0.45 es segura (valores de 1.0 a 2.0 son algo mejores, y 5.0 diverge).
5. La calidad y la **consistencia** de las etiquetas del profesor pesan más que cualquier hiperparámetro: DFS arruina el resultado aunque se imite bien.

Limitaciones de este análisis: 5 semillas y una única partición de datos; las diferencias de uno o dos puntos no son significativas. La evaluación usa hasta 400 pares de prueba con un límite de 60 intentos por partida.

## Alcance y limitaciones

El modelo no interpreta capturas de pantalla: trabaja con características estructuradas. Conoce el identificador de cada mapa. Los tres mapas pertenecen al mismo universo de entrenamiento. La evaluación demuestra generalización a pares de posición-meta no entrenados, no a laberintos totalmente desconocidos. Para eso habría que representar también las paredes de cualquier mapa nuevo y entrenar y evaluar en mapas separados.
