# Neurito 2.0: aprende a saltar obstáculos

Neurito aprende **cuándo saltar** ante obstáculos que se aproximan con diferentes velocidades. El cerebro se implementa manualmente con NumPy y el aprendizaje puede observarse en Pygame.

## ¿Qué aprende?

La red recibe tres características normalizadas: distancia al obstáculo, velocidad y tiempo estimado hasta alcanzarlo (distancia / velocidad). La última es una característica derivada, no una observación independiente.

La arquitectura es **3 → 12 → 1**: doce neuronas ocultas con ReLU y una salida con sigmoide. La última devuelve una puntuación entre cero y uno que, con el umbral 0.5, determina si Neurito debe saltar cuando está en el suelo.

## ¿Cómo aprende?

Aplicamos **aprendizaje supervisado**, no aprendizaje por refuerzo. Creamos datos sintéticos que incluyen ejemplos donde es conveniente saltar y otros donde debe esperar. Para generar las etiquetas, definimos una ventana de referencia: saltar cuando el tiempo estimado está entre **7 y 13 fotogramas**. No es una regla aprendida por recompensa ni garantiza evitar choques en todas las situaciones.

La red ajusta sus pesos a partir de la **entropía cruzada binaria**, la **retropropagación** y el **descenso de gradiente**:

`W_nuevo = W_anterior - tasa_de_aprendizaje * gradiente`

El conjunto de entrenamiento equilibra aproximadamente las dos decisiones. De otra manera, una red podría aparentar buena precisión diciendo casi siempre «esperar».

La precisión se mide sobre ejemplos distintos de los utilizados en el entrenamiento. **Una precisión alta no garantiza un buen resultado en el juego**, que también depende de la física del salto.

## Archivos

```text
Neurito/
├── main.py                 # Mascota original: comer, dormir, jugar
├── red-sencilla.py         # Ejemplo de una neurona
├── salto_modelo.py         # Red neuronal manual y datos de entrenamiento
├── salto_juego.py          # Física y detección de colisiones
├── saltar.py               # Juego y visualización del entrenamiento
├── GUIA_SALTO.md
└── assets/
    └── jugar_1.png        # Reutilizado en el nuevo juego
```

## Ejecución

```bash
uv sync
uv run saltar.py
```

No se necesitan dependencias nuevas: se reutilizan NumPy y Pygame.

## Controles

| Tecla | Función |
|---|---|
| `1` | Entrenar una época |
| `2` | Entrenar cien épocas |
| `3` | Activar o pausar entrenamiento continuo |
| `A` | Alternar control automático y manual |
| `Espacio` | Saltar manualmente |
| `C` | Reiniciar pista y contadores |
| `R` | Reiniciar pesos y contadores |
| `Esc` | Salir |

## Experimentos

1. Prueba el juego sin entrenar. Anota colisiones y obstáculos esquivados.
2. Entrena de cien en cien épocas; compara la pérdida, la precisión y el comportamiento.
3. Cambia `ocultas=12` por `ocultas=3` o `ocultas=24`; compara generalización.
4. Modifica la tasa de aprendizaje `0.8` a `0.01` o `1.2` y estudia la convergencia.
5. Cambia la ventana de entrenamiento en `etiqueta()` y en la generación de positivos. Comprueba si la política resultante es compatible con la física.

**Pregunta central:** ¿por qué una red puede clasificar ejemplos con buena precisión y aun así chocar?

## Resultados de los experimentos

**Método.** Los experimentos se ejecutaron sin interfaz gráfica, con las mismas clases del proyecto (`RedSalto` y `JuegoSalto`) y reproduciendo el bucle de `saltar.py` (probabilidad ≥ 0.5 y en el suelo → saltar). Cada medición simula **9000 fotogramas (5 min a 30 fps)**, con la misma secuencia de velocidades (semilla 19 del juego) y datos de entrenamiento con semilla 42 (5000 de entrenamiento y 1500 de prueba). «Esquivados» y «choques» son los contadores de la pantalla; el máximo posible en esa simulación es 74 esquivados con 0 choques. La «época 0» es la red sin entrenar. Los resultados son de una sola semilla, así que las diferencias pequeñas pueden cambiar con otra.

### Experimento 1: sin entrenar

| Épocas | Pérdida | Precisión (prueba) | Esquivados | Choques |
|---|---|---|---|---|
| 0 | sin entrenar | 44.9 % | 0 | 59 |

Con pesos aleatorios, la red tiene una probabilidad de salto casi constante que casi nunca llega a 0.5 en el momento correcto. Neurito no salta, choca con cada obstáculo (59 choques, uno cada ~5 s, porque tras el choque la pista se reinicia) y no esquiva ninguno. La precisión (44.9 %) es peor que la de responder siempre «esperar», que daría ~50 % en un conjunto equilibrado.

### Experimento 2: entrenar de cien en cien épocas

| Épocas | Pérdida | Precisión (prueba) | Esquivados | Choques |
|---|---|---|---|---|
| 0 | sin entrenar | 44.9 % | 0 | 59 |
| 100 | 0.2601 | 91.2 % | 67 | 5 |
| 200 | 0.2306 | 91.9 % | 72 | 1 |
| 300 | 0.2066 | 92.3 % | 74 | 0 |
| 400 | 0.1753 | 92.6 % | 74 | 0 |
| 500 | 0.1368 | 94.1 % | 74 | 0 |
| 1000 | 0.0574 | 97.8 % | 74 | 0 |
| 2000 | 0.0306 | 98.8 % | 74 | 0 |
| 5000 | 0.0144 | 99.6 % | 74 | 0 |

**Observaciones.**
- La mayor parte de la mejora ocurre en las primeras 100 épocas: la precisión salta de 44.9 % a 91.2 % y Neurito ya esquiva 67 de 72 obstáculos.
- A las 300 épocas el comportamiento es perfecto (0 choques), aunque la precisión solo es de 92.3 %. A partir de ahí, la pérdida y la precisión siguen mejorando, pero **el comportamiento en el juego ya no cambia**: no hay más que ganar en el juego.
- La pérdida baja de forma continua (0.26 → 0.014), con una meseta inicial y una bajada más rápida entre 400 y 1000 épocas, cuando la red afina los bordes de la ventana de 7 a 13 fotogramas.

### Experimento 3: número de neuronas ocultas

| Ocultas | Épocas | Pérdida | Precisión (prueba) | Esquivados | Choques |
|---|---|---|---|---|---|
| 3 | 100 | 0.2649 | 90.3 % | 60 | 11 |
| 3 | 500 | 0.2153 | 92.9 % | 74 | 0 |
| 3 | 1000 | 0.2060 | 93.2 % | 74 | 0 |
| 3 | 3000 | 0.1978 | 93.7 % | 74 | 0 |
| 12 | 100 | 0.2601 | 91.2 % | 67 | 5 |
| 12 | 500 | 0.1368 | 94.1 % | 74 | 0 |
| 12 | 1000 | 0.0574 | 97.8 % | 74 | 0 |
| 12 | 3000 | 0.0219 | 99.1 % | 74 | 0 |
| 24 | 100 | 0.2629 | 90.8 % | 67 | 5 |
| 24 | 500 | 0.0932 | 96.4 % | 74 | 0 |
| 24 | 1000 | 0.0502 | 97.7 % | 74 | 0 |
| 24 | 3000 | 0.0207 | 99.3 % | 74 | 0 |

**Observaciones.**
- Con **3 neuronas** la red tiene poca capacidad: la pérdida se estanca en ~0.20 y la precisión en ~93.7 %. No logra representar bien una ventana con dos bordes (saltar solo si 7 ≤ tiempo ≤ 13), por lo que los errores se concentran en los bordes. Aun así, en el juego esquiva todo a partir de las 500 épocas.
- Con **12 y 24 neuronas** la red recoge los dos bordes y llega a 99 %. 24 converge más rápido (a 500 épocas: 96.4 % frente a 94.1 %), pero a 3000 épocas ambas quedan prácticamente iguales (99.1 % frente a 99.3 %). Las 12 neuronas del proyecto son suficientes.
- La precisión de prueba se mide con datos distintos a los de entrenamiento y es similar a la de entrenamiento, así que no se observó sobreajuste. El problema es sencillo (tres entradas y una regla), por lo que más capacidad no aporta una generalización notablemente mejor, solo mayor velocidad.
- Con 24 neuronas y sin entrenar, la inicialización aleatoria dio una precisión de 58.2 % y 24 choques (frente a 59), lo que muestra que el resultado sin entrenar depende de la semilla.

### Experimento 4: tasa de aprendizaje

| Tasa | Épocas | Pérdida | Precisión (prueba) | Esquivados | Choques |
|---|---|---|---|---|---|
| 0.01 | 100 | 0.6449 | 90.7 % | 58 | 12 |
| 0.01 | 500 | 0.5149 | 85.9 % | 0 | 56 |
| 0.01 | 1000 | 0.4302 | 86.5 % | 0 | 56 |
| 0.01 | 3000 | 0.3111 | 88.6 % | 0 | 55 |
| 0.8 | 100 | 0.2601 | 91.2 % | 67 | 5 |
| 0.8 | 500 | 0.1368 | 94.1 % | 74 | 0 |
| 0.8 | 1000 | 0.0574 | 97.8 % | 74 | 0 |
| 0.8 | 3000 | 0.0219 | 99.1 % | 74 | 0 |
| 1.2 | 100 | 0.2560 | 89.3 % | 57 | 13 |
| 1.2 | 500 | 0.0855 | 95.3 % | 74 | 0 |
| 1.2 | 1000 | 0.0392 | 98.3 % | 74 | 0 |
| 1.2 | 3000 | 0.0157 | 99.6 % | 74 | 0 |

**Observaciones.**
- **0.01 (demasiado pequeña):** la pérdida baja muy lentamente (0.64 → 0.31 en 3000 épocas). Lo más llamativo es que **la precisión y el juego empeoran entre las épocas 100 y 500**: la red pasa de saltar a no saltar nunca (0 esquivados, 56 choques) mientras la pérdida sí mejora. La pérdida mide la calidad de las probabilidades, no si cruzan 0.5; la red aprende primero a bajar la probabilidad en los muchos ejemplos «esperar» y la probabilidad máxima de los casos «saltar» sigue por debajo de 0.5. Con 3000 épocas todavía no sirve para jugar.
- **0.8 (valor del proyecto):** converge de forma estable y buena.
- **1.2 (mayor):** en este problema no diverge ni oscila; converge más rápido a partir de unas 300 épocas y llega a la menor pérdida (0.0157). Al inicio parece algo peor (89.3 % a las 100 épocas, 13 choques) por los pasos grandes, pero la pérdida baja sin saltos. Con un problema más difícil o sin normalizar las entradas, una tasa tan alta podría oscilar o diverger. Aquí no se ha observado.
- Conclusión: una tasa demasiado baja necesita muchísimas más épocas (y puede dar la falsa impresión de progreso porque la pérdida baja), y una tasa grande acelera siempre que siga siendo estable.

### Experimento 5: cambiar la ventana de entrenamiento

Se modificaron a la vez `etiqueta()` y los ejemplos positivos de `generar_datos()` (que se generan unos 0.5 fotogramas dentro de cada borde de la ventana). Los archivos del proyecto no se modificaron: se usó una subclase en el script de pruebas. Después se midió la física real: dónde puede Neurito saltar y esquivar el obstáculo, con el salto lanzado cuando el tiempo estimado hasta el obstáculo (distancia / velocidad) es `t`.

**Física real.** El salto dura 36 fotogramas en el aire. Con el salto lanzado en el mismo fotograma, el obstáculo se esquiva si el tiempo estimado está entre unos **4 fotogramas** y un máximo que depende de la velocidad: **14.75 (v = 4.5), 18 (5.5), 20 (6.5), 21.5 (7.5) y 22.75 (8.5)**. Si se salta antes de ese máximo, Neurito aterriza antes de pasar el obstáculo y choca; si se salta con menos de 4, el obstáculo ya está encima. La ventana segura para todas las velocidades es de **4 a 14.75 fotogramas**, y la de referencia (7 a 13) queda dentro.

| Ventana | Épocas | Pérdida | Precisión (prueba) | Esquivados | Choques | ¿Compatible con la física? |
|---|---|---|---|---|---|---|
| 7–13 (original) | 500 | 0.1368 | 94.1 % | 74 | 0 | Sí, dentro de 4–14.75 |
| 7–13 (original) | 2000 | 0.0306 | 98.8 % | 74 | 0 | Sí |
| 20–30 | 500 | 0.0676 | 97.7 % | 0 | 59 | **No**, demasiado pronto |
| 20–30 | 2000 | 0.0258 | 99.1 % | 0 | 59 | **No** |
| 14–22 | 500 | 0.0784 | 97.5 % | 0 | 56 | **No** (solo vale para velocidades altas) |
| 14–22 | 2000 | 0.0275 | 99.0 % | 0 | 56 | **No** |
| 1–4 | 500 | 0.0745 | 97.3 % | 74 | 0 | Casi: en el límite inferior |
| 1–4 | 2000 | 0.0564 | 98.4 % | 54 | 17 | **Parcialmente** (muy tarde) |
| 3–7 | 500 | 0.1395 | 96.5 % | 74 | 0 | Sí |
| 3–7 | 2000 | 0.0959 | 97.1 % | 74 | 0 | Sí |

**Observaciones.**
- Las ventanas **20–30** y **14–22** consiguen una precisión aún mayor (99 %) y **chocan siempre** (0 esquivados): la red aprende a la perfección una regla que no sirve para esquivar, porque salta demasiado pronto y aterriza antes de que pase el obstáculo. La 14–22 solo funcionaría con la velocidad más alta.
- La ventana **1–4** está en el límite de lo posible: a 500 épocas funciona, pero a 2000 épocas, cuando la red afina más sus bordes, el umbral se desplaza hacia valores menores de 4 y aparecen 17 choques, porque Neurito salta demasiado tarde.
- La ventana **3–7** es compatible con la física, y funciona aunque salte más tarde que la original, pero con poco margen frente al límite inferior de 4.
- La **precisión no predice el comportamiento**: la ventana peor para el juego (20–30) tiene la mejor precisión, y la ventana original, que es la buena, tiene menos.

## Respuesta a la pregunta central

**¿Por qué una red puede clasificar ejemplos con buena precisión y aun así chocar?**

Porque la precisión solo mide si la red **imita las etiquetas**, no si esas etiquetas son una buena política de juego. Los experimentos muestran varias razones:

1. **Las etiquetas pueden ser incompatibles con la física.** Con la ventana 20–30, la red obtuvo 99.1 % de precisión y 59 choques sobre 59 obstáculos. Aprendió perfectamente una regla equivocada: saltar con 20 a 30 fotogramas de margen hace que Neurito caiga antes de que llegue el obstáculo. La red no conoce la gravedad ni las colisiones, solo los ejemplos que se le dan.
2. **La precisión se promedia sobre todos los ejemplos, pero el juego depende de uno solo.** Basta un solo error en el instante crítico (saltar un par de fotogramas tarde o pronto) para chocar, y el juego exige acertar todas las veces, no el 99 %. Además, con la tasa 0.01 la precisión llegó a 86–88 % (la mayoría de los ejemplos son «esperar») y la red no saltaba nunca.
3. **Los errores se concentran donde más importan.** Las equivocaciones de la red son cerca de los bordes de la ventana (7 y 13 fotogramas), justo donde se decide el salto.
4. **Se evalúa de forma distinta que en el juego.** La precisión se mide sobre ejemplos independientes con distancias y velocidades repartidas, mientras que en el juego la red decide en cada fotograma y en cuanto cruza 0.5 salta, y no puede corregir un salto hecho antes de tiempo.5. **La pérdida y la precisión no son la métrica de la tarea.** Con la tasa 0.01 la pérdida bajaba mientras el comportamiento empeoraba (de 58 esquivados a 0), y con 3 neuronas la precisión se estancó en 93.7 % pero se esquivó todo. Entre las épocas 300 y 5000 la precisión pasó de 92.3 % a 99.6 % sin ningún cambio en el juego.

**Conclusión:** hay que evaluar un modelo en el entorno real (esquivados y choques), no solo con la precisión sobre los datos de prueba, y hay que comprobar que las etiquetas que se usan para entrenar reflejan realmente una buena política según la física.
