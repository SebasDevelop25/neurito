# Neurito: experimentos y resultados

Todos los resultados se generaron con `experimentos.py`, que usa la misma red, los mismos datos y el mismo entrenamiento que `main.py` (3 entradas → 6 neuronas ReLU → 3 salidas Softmax, descenso de gradiente por lotes completos, 1200 épocas, tasa de aprendizaje 0.1, semilla 42), pero sin la interfaz gráfica. Para reproducirlos:

```
uv run python experimentos.py
```

**Política sintética que la red debe imitar:** se calcula `[hambre, 1 − energía, aburrimiento]` y se elige la acción del valor más alto (COMER, DORMIR o JUGAR).

**Línea base:**

| época | 0 | 200 | 400 | 600 | 800 | 1000 | final (1199) |
|---|---|---|---|---|---|---|---|
| pérdida | 1.0742 | 0.4961 | 0.2644 | 0.1942 | 0.1592 | 0.1375 | 0.1225 |

Precisión en 1000 ejemplos de prueba: **99.40 %**.

---

## Experimentos

### 1. Dos necesidades fijas, la tercera variable: ¿cuándo cambia la acción?

**a) Variando el aburrimiento** (hambre = 0.3, energía = 0.6, es decir, cansancio = 0.4)

| aburrimiento | red | política | COMER | DORMIR | JUGAR |
|---|---|---|---|---|---|
| 0.0 | DORMIR | DORMIR | 13.1 % | 86.9 % | 0.0 % |
| 0.2 | DORMIR | DORMIR | 13.2 % | 85.1 % | 1.7 % |
| 0.3 | DORMIR | DORMIR | 12.2 % | 76.9 % | 10.9 % |
| **0.4** | **JUGAR** | DORMIR | 7.5 % | **46.2 %** | **46.3 %** |
| 0.5 | JUGAR | JUGAR | 2.0 % | 12.1 % | 85.9 % |
| 0.7 | JUGAR | JUGAR | 0.0 % | 0.3 % | 99.7 % |
| 1.0 | JUGAR | JUGAR | 0.0 % | 0.0 % | 100.0 % |

La red pasa de DORMIR a JUGAR con aburrimiento ≈ **0.400** y la política con ≈ **0.401**. El cambio ocurre justo cuando el aburrimiento supera al cansancio (0.4). En el punto exacto de empate la red reparte casi 50/50 (46.2 % frente a 46.3 %).

**b) Variando el hambre** (energía = 0.6, es decir, cansancio = 0.4; aburrimiento = 0.3)

| hambre | red | política | COMER | DORMIR | JUGAR |
|---|---|---|---|---|---|
| 0.0 | DORMIR | DORMIR | 0.0 % | 85.3 % | 14.7 % |
| 0.3 | DORMIR | DORMIR | 12.2 % | 76.9 % | 10.9 % |
| **0.4** | **COMER** | COMER | **47.6 %** | **46.3 %** | 6.1 % |
| 0.5 | COMER | COMER | 85.5 % | 12.9 % | 1.6 % |
| 1.0 | COMER | COMER | 100.0 % | 0.0 % | 0.0 % |

Cambio de DORMIR a COMER: la red en 0.399 y la política en 0.400, de nuevo cuando el hambre alcanza al cansancio.

**c) Variando la energía** (hambre = 0.3, aburrimiento = 0.3)

| energía | cansancio | red | política | COMER | DORMIR | JUGAR |
|---|---|---|---|---|---|---|
| 0.0 | 1.0 | DORMIR | DORMIR | 0.0 % | 100.0 % | 0.0 % |
| 0.4 | 0.6 | DORMIR | DORMIR | 0.4 % | 99.3 % | 0.3 % |
| 0.6 | 0.4 | DORMIR | DORMIR | 12.2 % | 76.9 % | 10.9 % |
| **0.7** | **0.3** | **COMER** | DORMIR | **34.9 %** | **33.6 %** | **31.6 %** |
| 0.8 | 0.2 | COMER | COMER | 48.4 % | 7.1 % | 44.5 % |
| 1.0 | 0.0 | COMER | COMER | 51.3 % | 0.2 % | 48.5 % |

La acción cambia cuando el cansancio baja de 0.3, el valor de las otras dos necesidades (red: energía 0.699; política: 0.700). Hay un detalle: al bajar el cansancio, COMER y JUGAR quedan empatados (hambre = aburrimiento = 0.3). Por eso la red nunca pasa de ~51 % frente a ~48 %: no tiene forma de distinguir entre dos acciones con la misma prioridad.

**Conclusión:** la acción cambia exactamente cuando la necesidad variable **supera a la mayor de las otras dos**. La red aprendió la frontera con un error de ~0.001. La probabilidad no salta de golpe: forma una transición suave de unas ±0.1 alrededor de la frontera.

### 2. Prioridades casi iguales: ¿cómo cambia Softmax?

| hambre | energía (cansancio) | aburrimiento | política | red | COMER | DORMIR | JUGAR |
|---|---|---|---|---|---|---|---|
| 0.50 | 0.50 (0.50) | 0.50 | COMER* | COMER | 35.2 % | 33.4 % | 31.3 % |
| 0.51 | 0.50 (0.50) | 0.50 | COMER | COMER | 39.7 % | 31.3 % | 29.1 % |
| 0.50 | 0.49 (0.51) | 0.50 | DORMIR | DORMIR | 32.9 % | 37.8 % | 29.3 % |
| 0.50 | 0.50 (0.50) | 0.51 | JUGAR | JUGAR | 33.0 % | 31.3 % | 35.7 % |
| 0.80 | 0.20 (0.80) | 0.80 | COMER* | COMER | 35.7 % | 33.3 % | 31.0 % |
| 0.81 | 0.20 (0.80) | 0.79 | COMER | COMER | 42.3 % | 32.7 % | 24.9 % |
| 0.20 | 0.80 (0.20) | 0.20 | COMER* | COMER | 34.7 % | 33.6 % | 31.7 % |
| 0.90 | 0.10 (0.90) | 0.10 | COMER* | DORMIR | 48.8 % | 51.2 % | 0.0 % |
| 0.90 | 0.90 (0.10) | 0.10 | COMER | COMER | 100.0 % | 0.0 % | 0.0 % |

\* Empate exacto: `argmax` elige la primera acción (COMER) por convención, no por una razón real.

Observaciones:

- Con prioridades iguales Softmax da salidas **casi uniformes (~33 % cada una)**. La red no tiene una preferencia clara.
- Basta una diferencia de **0.01** en una necesidad para que esa acción gane, pero solo por **4 a 7 puntos porcentuales**. La acción cambia aunque la confianza sigue siendo baja.
- Softmax depende de la **diferencia** entre prioridades, no de su valor absoluto: `[0.5, 0.5, 0.5]`, `[0.8, 0.2, 0.8]` y `[0.2, 0.8, 0.2]` dan casi la misma distribución. En cambio `[0.9, 0.9, 0.1]` da 100 % COMER porque el hambre (0.9) le gana al cansancio (0.1) por 0.8.
- En los empates exactos la red puede elegir otra cosa que la política (`[0.9, 0.1, 0.1]`: DORMIR 51.2 % frente a COMER 48.8 %). No es un error grave: la etiqueta de la política en ese punto es arbitraria.

### 3. Tamaño de la capa oculta: 3, 6, 10 o 20 neuronas

**Con la semilla 42 (la misma de `main.py`):**

| neuronas | pérdida final | precisión test |
|---|---|---|
| 3 | 0.1618 | 96.30 % |
| 6 | 0.1225 | 99.40 % |
| 10 | 0.1262 | 99.30 % |
| 20 | 0.1220 | 98.80 % |

**Promedio de 5 semillas (0 a 4), para no depender de una sola inicialización:**

| neuronas | pérdida media | precisión media | precisión mín – máx |
|---|---|---|---|
| 3 | 0.2768 | 88.88 % | **60.80 %** – 96.40 % |
| 6 | 0.1534 | 97.78 % | 96.80 % – 99.20 % |
| 10 | 0.1222 | 98.98 % | 98.50 % – 99.30 % |
| 20 | 0.1207 | 99.18 % | 98.80 % – 99.50 % |

Observaciones:

- **3 neuronas** es claramente insuficiente e inestable: según la inicialización puede quedarse en 60.8 %. Con tan pocas neuronas, si una queda "muerta" (ReLU siempre en 0), la red pierde una gran parte de su capacidad.
- **De 6 a 10 neuronas** mejoran la pérdida y la estabilidad entre semillas.
- **De 10 a 20** la mejora es mínima (0.1222 → 0.1207). El problema es simple (tres comparaciones) y más neuronas no aportan mucho.
- Con una sola semilla parecía que 20 neuronas eran peores que 6 (98.8 % frente a 99.4 %). El promedio muestra que fue **ruido de la inicialización**. Por eso conviene repetir los experimentos con varias semillas.

### 4. Épocas y tasa de aprendizaje

**a) Número de épocas (lr = 0.1)**

| épocas | pérdida final | precisión test |
|---|---|---|
| 100 | 0.8088 | 77.30 % |
| 300 | 0.3397 | 95.90 % |
| 1200 | 0.1225 | 99.40 % |
| 5000 | 0.0535 | 99.60 % |
| 20000 | 0.0255 | 99.70 % |

Con más épocas el entrenamiento **mejora y luego se estanca**: la pérdida sigue bajando, pero la precisión casi no sube después de 1200 épocas. Lo que gana la red es más confianza en las mismas respuestas, no más aciertos. No hay sobreajuste porque los datos de prueba vienen de la misma política sin ruido.

**b) Tasa de aprendizaje (1200 épocas)**

| lr | pérdida ép. 0 | ép. 300 | ép. 600 | final | veces que la pérdida subió | neuronas muertas | precisión | comportamiento |
|---|---|---|---|---|---|---|---|---|
| 0.001 | 1.0742 | 1.0624 | 1.0520 | 1.0333 | 0 | 0/6 | 41.80 % | demasiado lento |
| 0.01 | 1.0742 | 0.9848 | 0.9119 | 0.7376 | 0 | 0/6 | 82.90 % | lento, sin terminar |
| **0.1** | 1.0742 | 0.3387 | 0.1942 | 0.1225 | 0 | 0/6 | 99.40 % | estable (la de `main.py`) |
| **0.5** | 1.0742 | 0.1068 | 0.0711 | 0.0484 | 0 | 0/6 | **99.60 %** | estable y más rápido |
| 1.0 | 1.0742 | 0.1720 | 0.0888 | 0.0447 | 545 | 0/6 | 98.70 % | **oscila** (subidas de hasta 0.28) |
| 5.0 | 1.0742 | 1.0980 | 1.0980 | 1.0979 | 8 | 5/6 | 33.80 % | **se estanca** en ln 3 ≈ 1.0986 |
| 20.0 | 1.0742 | 9.0175 | 9.0175 | 6.1621 | 800 | 6/6 | 33.80 % | **diverge** |

Observaciones:

- **lr muy pequeña (0.001 – 0.01):** la pérdida baja tan despacio que 1200 épocas no alcanzan.
- **lr 0.1 – 0.5:** zona buena. La pérdida baja en todas las épocas sin subir nunca.
- **lr 1.0:** la pérdida sube 545 veces de una época a la siguiente. Los pasos son tan grandes que la red "rebota" alrededor del mínimo. Aun así llega a un resultado aceptable.
- **lr 5.0:** un primer paso enorme deja **5 de las 6 neuronas ReLU muertas** (su salida es 0 para todas las entradas, así que su gradiente también es 0 y ya no se recuperan). La red queda dando ~33 % a cada acción: pérdida = ln 3, que es la pérdida de adivinar al azar.
- **lr 20.0:** la pérdida explota (~9) y las 6 neuronas mueren. Los 33.8 % de precisión equivalen a predecir siempre la misma clase.

### 5. ¿Qué datos y etiquetas requeriría una cuarta acción?

Ejemplo: una acción **BAÑARSE** asociada a una necesidad nueva, **suciedad** (0 = limpio, 1 = muy sucio).

1. **Entrada:** agregar una cuarta columna a `X` (`np.random.rand(3000, 4)`) y la barra correspondiente en la interfaz.
2. **Etiquetas:** ampliar la política para que considere la nueva prioridad:
   ```python
   targets = np.argmax(np.column_stack((X[:, 0], 1 - X[:, 1], X[:, 2], X[:, 3])), axis=1)
   ```
   Las etiquetas pasan a ser 0, 1, 2 o 3.
3. **Red:** `W1` pasa a ser de 4×N y `W2` y `b2` a ser de N×4 y 1×4 (cuatro salidas Softmax). La tupla `actions` necesita `"bathe"`, más textos y animaciones nuevas.
4. **Datos suficientes y balanceados:** con 4 clases uniformes cada acción recibe ~25 % de los ejemplos (750 de 3000). Si la nueva acción fuera rara, habría que generar más ejemplos de ella o ponderarla. Si no, la red podría ignorarla y aun así tener una precisión alta.
5. **Casos frontera:** incluir ejemplos donde la suciedad compite con las otras necesidades, para que la red aprenda dónde está la frontera nueva.
6. **Alternativa sin entrada nueva:** si la cuarta acción dependiera de una combinación de las necesidades actuales (por ejemplo, "DESCANSAR" cuando todas son bajas, como el estado NEUTRAL que hoy es una regla fija en `main.py`), no haría falta otra entrada, pero sí **etiquetas nuevas** para esa región. Probablemente también más neuronas, porque la frontera ya no sería un simple "cuál es el mayor".

---

## Explicación de los resultados

### ¿En qué se diferencian pesos y sesgos?

- **Pesos (`W1`, `W2`):** multiplican las entradas. Indican **cuánto y en qué dirección** influye cada entrada en cada neurona. Por ejemplo, en la red entrenada la columna 4 de `W1` vale `[2.361, −0.352, −2.170]`: esa neurona se activa con el hambre y se inhibe con el aburrimiento, así que funciona como un detector de "hambre > aburrimiento".
- **Sesgos (`b1`, `b2`):** se suman sin depender de la entrada. **Desplazan el umbral** de activación: indican cuánto se activa una neurona cuando todas las entradas valen 0. El sesgo aprendido `b2 = [−0.982, 1.030, −0.048]` favorece DORMIR. Con la entrada `[0, 0, 0]`, donde solo actúan los sesgos y la energía 0 significa cansancio máximo, la red da **DORMIR 100 %**.
- En geometría: los pesos definen la **orientación** de la frontera de decisión y los sesgos su **posición**. Sin sesgos, todas las fronteras tendrían que pasar por el origen.

### ¿Por qué el cansancio es 1 − energía?

La política elige "la necesidad más urgente", así que las tres entradas deben medirse en la misma dirección: **más alto = más urgente**. El hambre y el aburrimiento ya cumplen eso, pero la energía es al revés: energía alta significa que **no** hace falta dormir. Al invertirla (`1 − energía`) se obtiene el cansancio, también entre 0 y 1, y las tres necesidades se pueden comparar con un solo `argmax`. El experimento 1c lo muestra: al **subir** la energía baja la probabilidad de DORMIR.

La red recibe la energía sin invertir y aprende sola esa inversión con pesos negativos. Por ejemplo, la columna 2 de `W1` tiene un peso de −0.835 en la energía.

### ¿Qué hace ReLU y qué hace Softmax?

- **ReLU** (`max(0, x)`, en la capa oculta): deja pasar los valores positivos y convierte los negativos en 0. Introduce **no linealidad**: sin ella, las dos capas se reducirían a una sola transformación lineal. Con ella cada neurona oculta solo "se enciende" en una región del espacio de entradas, como el detector "hambre > aburrimiento" de antes. Su riesgo es que una neurona que siempre da 0 deja de aprender (**neurona muerta**), y eso fue lo que pasó con lr = 5 y lr = 20.
- **Softmax** (en la salida): convierte los tres puntajes en **probabilidades positivas que suman 1**, con `exp(zᵢ) / Σ exp(zⱼ)`. Depende de la **diferencia** entre puntajes, no de su valor absoluto: por eso en el experimento 2 los empates dan ~33 % y una ventaja pequeña da apenas 4 a 7 puntos más. Junto con la entropía cruzada da el gradiente simple `probabilidades − etiqueta` que usa el código (`delta2`).

### ¿Por qué una precisión alta puede ser engañosa?

El 99.4 % de la línea base es real, pero oculta varias cosas:

1. **Los errores se concentran cerca de las fronteras.** Si se mide la ventaja de la necesidad ganadora sobre la segunda en los datos de prueba:

   | ventaja de la prioridad ganadora | ejemplos | precisión |
   |---|---|---|
   | < 0.02 | 60 | **90.0 %** |
   | 0.02 – 0.05 | 91 | 100 % |
   | 0.05 – 0.10 | 114 | 100 % |
   | > 0.10 | 735 | 100 % |

   Todos los errores (6 de 1000) están en casos casi empatados. La mayoría de los ejemplos son fáciles y eso infla el promedio.

2. **La precisión no mide la confianza.** En los aciertos la confianza media es 89.8 %; en los errores, 51.1 %. Una red podría acertar con 34 % de probabilidad y contaría igual que una que acierta con 100 %. La **pérdida** sí refleja esa diferencia, y por eso siguió bajando con más épocas aunque la precisión casi no cambiara.

3. **Depende del balance de clases.** Aquí las clases están balanceadas (338 / 320 / 342), así que el mínimo trivial es ~34 % (predecir siempre la clase mayoritaria). Si el 95 % de los datos fueran COMER, un modelo que siempre diga COMER tendría 95 % sin aprender nada. La matriz de confusión (filas = política, columnas = red) muestra que aquí no pasa eso:

   |  | COMER | DORMIR | JUGAR |
   |---|---|---|---|
   | **COMER** | 338 | 0 | 0 |
   | **DORMIR** | 0 | 316 | 4 |
   | **JUGAR** | 2 | 0 | 340 |

4. **Solo mide la imitación de la política sintética.** La red copia una regla que inventamos. Si la regla fuera mala, la red sería "99 % precisa" en hacer algo malo. Además, la animación final usa una regla fija para el estado NEUTRAL que la red nunca aprendió.

5. **Fuera de rango no hay garantías.** Con valores fuera de [0, 1], que nunca vio en el entrenamiento, la red responde con total seguridad: hambre = 2 da COMER 100 % y aburrimiento = 3 da JUGAR 100 %. Esos casos son imposibles en la política y la precisión de prueba no dice nada sobre ellos.

6. **Depende de la semilla.** En el experimento 3 la misma arquitectura de 3 neuronas dio entre 60.8 % y 96.4 % según la inicialización. Un solo número no basta para sacar conclusiones.
