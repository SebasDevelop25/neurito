"""Ejecuta los experimentos de GUIA_LABERINTO.md sin Pygame y muestra tablas de resultados.

Uso: uv run python experimentos_laberinto.py [numero_experimento ...]
"""
import heapq
import sys
import numpy as np
from laberinto_mundo import (MAPAS, libres, vecino, valido, distancias_bfs,
                             accion_experta)
from laberinto_red import RedLaberinto, codificar

SEMILLAS = (0, 1, 2, 3, 4)
MAX_PASOS = 60


# ---------- profesores (algoritmos de búsqueda que generan las etiquetas) ----------
def primer_paso(camino):
    return camino[1] if len(camino) > 1 else None


def accion_hacia(origen, destino):
    for a in range(4):
        if vecino(origen, a) == destino:
            return a


def camino_dfs(mapa, origen, meta):
    """DFS iterativo: encuentra un camino cualquiera (no necesariamente corto)."""
    pila = [(origen, [origen])]
    visto = {origen}
    while pila:
        pos, ruta = pila.pop()
        if pos == meta:
            return ruta
        for a in reversed(range(4)):
            sig = vecino(pos, a)
            if valido(mapa, sig) and sig not in visto:
                visto.add(sig)
                pila.append((sig, ruta + [sig]))


def camino_astar(mapa, origen, meta):
    h = lambda p: abs(p[0] - meta[0]) + abs(p[1] - meta[1])
    abierta = [(h(origen), 0, origen)]
    padre = {origen: None}
    g = {origen: 0}
    while abierta:
        _, costo, pos = heapq.heappop(abierta)
        if pos == meta:
            ruta = []
            while pos is not None:
                ruta.append(pos)
                pos = padre[pos]
            return ruta[::-1]
        for a in range(4):
            sig = vecino(pos, a)
            if valido(mapa, sig) and costo + 1 < g.get(sig, 10**9):
                g[sig] = costo + 1
                padre[sig] = pos
                heapq.heappush(abierta, (costo + 1 + h(sig), costo + 1, sig))


def etiqueta(profesor, mapa, pos, meta, dist):
    if profesor == 'BFS':
        return accion_experta(mapa, pos, dist)
    ruta = (camino_dfs if profesor == 'DFS' else camino_astar)(mapa, pos, meta)
    return accion_hacia(pos, primer_paso(ruta))


def datos(profesor='BFS', semilla=42, fraccion_prueba=0.2):
    muestras = []
    for i, mapa in enumerate(MAPAS):
        celdas = libres(mapa)
        for meta in celdas:
            dist = distancias_bfs(mapa, meta)
            for pos in celdas:
                if pos != meta and pos in dist:
                    muestras.append((i, pos, meta, etiqueta(profesor, mapa, pos, meta, dist)))
    rng = np.random.default_rng(semilla)
    rng.shuffle(muestras)
    corte = int(len(muestras) * (1 - fraccion_prueba))
    def arr(f):
        return (np.stack([codificar(i, p, m) for i, p, m, a in f]),
                np.asarray([a for *_, a in f]))
    return arr(muestras[:corte]), arr(muestras[corte:]), muestras[corte:]


# ---------- evaluación jugando ----------
def jugar(red, i, pos, meta, asistido):
    """Devuelve (éxito, pasos, choques, pasos_óptimos)."""
    mapa = MAPAS[i]
    optimo = distancias_bfs(mapa, meta)[pos]
    visitas = {pos: 1}
    pasos = choques = 0
    for _ in range(MAX_PASOS):
        probs = red.probabilidades(i, pos, meta)
        if asistido:
            ops = [(float(probs[a]) - 0.35 * visitas.get(vecino(pos, a), 0), -a, a)
                   for a in range(4) if valido(mapa, vecino(pos, a))]
            a = max(ops)[2]
        else:
            a = int(probs.argmax())
        nuevo = vecino(pos, a)
        if not valido(mapa, nuevo):
            choques += 1
            continue                       # se queda en su sitio, como en el juego
        pos = nuevo
        pasos += 1
        visitas[pos] = visitas.get(pos, 0) + 1
        if pos == meta:
            return True, pasos, choques, optimo
    return False, pasos, choques, optimo


def evaluar(red, prueba, asistido, maximo=400):
    r = [jugar(red, i, p, m, asistido) for i, p, m, _ in prueba[:maximo]]
    ex = np.array([x[0] for x in r])
    exito = ex.mean()
    pasos = np.mean([x[1] for x in r if x[0]]) if ex.any() else float('nan')
    extra = np.mean([x[1] / x[3] for x in r if x[0]]) if ex.any() else float('nan')
    choques = np.mean([x[2] for x in r])
    return exito, pasos, extra, choques


def entrenar(epocas, semilla, ocultas=96, tasa=0.45, profesor='BFS'):
    (X, y), (Xp, yp), prueba = datos(profesor)
    red = RedLaberinto(semilla=semilla, ocultas=ocultas, tasa=tasa)
    for _ in range(epocas):
        red.entrenar_epoca(X, y)
    return red, X, y, Xp, yp, prueba


def fila(red, X, y, Xp, yp, prueba):
    """Pérdida (train), exactitud train/test, éxito puro/asistido."""
    perd = red.historial[-1] if red.historial else float('nan')
    puro = evaluar(red, prueba, False)
    asis = evaluar(red, prueba, True)
    return perd, red.exactitud(X, y), red.exactitud(Xp, yp), puro, asis


def media(filas):
    """Promedia una lista de tuplas anidadas."""
    def m(vals):
        if isinstance(vals[0], tuple):
            return tuple(m([v[k] for v in vals]) for k in range(len(vals[0])))
        return float(np.nanmean(vals))
    return m(filas)


def imprimir_cabecera():
    print(f"{'config':<16}{'pérdida':>9}{'acc.tr':>8}{'acc.te':>8} |"
          f"{'PURA éxito':>11}{'choq':>6}{'extra':>7} |"
          f"{'ASIST éxito':>12}{'choq':>6}{'extra':>7}")


def imprimir(nombre, f):
    perd, atr, ate, puro, asis = f
    print(f"{nombre:<16}{perd:9.4f}{atr:8.3f}{ate:8.3f} |"
          f"{puro[0]:11.3f}{puro[3]:6.1f}{puro[2]:7.2f} |"
          f"{asis[0]:12.3f}{asis[3]:6.1f}{asis[2]:7.2f}")


def corrida(nombre, epocas, **kw):
    fs = []
    for s in SEMILLAS:
        red, X, y, Xp, yp, prueba = entrenar(epocas, s, **kw)
        fs.append(fila(red, X, y, Xp, yp, prueba))
    imprimir(nombre, media(fs))


# ---------- experimentos ----------
def exp1():
    print("\n== 1. Sin entrenar (0 épocas) ==")
    imprimir_cabecera()
    fs = []
    for s in SEMILLAS:
        (X, y), (Xp, yp), prueba = datos()
        red = RedLaberinto(semilla=s)
        fs.append(fila(red, X, y, Xp, yp, prueba))
    imprimir("0 épocas", media(fs))
    # diagnóstico del atasco: ¿qué acciones elige la red sin entrenar?
    red = RedLaberinto(semilla=0)
    (_, _), (_, _), prueba = datos()
    elegidas = np.bincount([int(red.probabilidades(i, p, m).argmax()) for i, p, m, _ in prueba], minlength=4)
    print("  acciones elegidas sin entrenar [ARRIBA ABAJO IZQ DER]:", elegidas.tolist())
    print("  acciones correctas del experto                      :",
          np.bincount([a for *_, a in prueba], minlength=4).tolist())
    print(f"  probabilidad media máxima (confianza): "
          f"{np.mean([red.probabilidades(i, p, m).max() for i, p, m, _ in prueba]):.3f}")


def exp2():
    print("\n== 2. Épocas de entrenamiento (media de 5 semillas) ==")
    imprimir_cabecera()
    for e in (0, 100, 200, 400):
        corrida(f"{e} épocas", e)


def exp3():
    print("\n== 3. Pura vs asistida (ver columnas PURA / ASIST arriba) ==")
    imprimir_cabecera()
    corrida("400 épocas", 400)
    print("  La diferencia de éxito entre PURA y ASIST es lo que aportan las reglas, no la red.")


def exp4():
    print("\n== 4. Neuronas ocultas (400 épocas) ==")
    imprimir_cabecera()
    for n in (16, 32, 96, 128):
        corrida(f"{n} ocultas", 400, ocultas=n)


def exp5():
    print("\n== 5. Tasa de aprendizaje (400 épocas) ==")
    imprimir_cabecera()
    for t in (0.01, 0.1, 0.45, 1.0, 2.0, 5.0):
        corrida(f"tasa {t}", 400, tasa=t)


def exp6():
    print("\n== 6. Por mapa (400 épocas, evaluación en pares NO entrenados) ==")
    print(f"{'mapa':<8}{'acc.te':>8}{'PURA éxito':>12}{'ASIST éxito':>13}")
    acum = {i: [] for i in range(len(MAPAS))}
    for s in SEMILLAS:
        red, X, y, Xp, yp, prueba = entrenar(400, s)
        for i in range(len(MAPAS)):
            sub = [m for m in prueba if m[0] == i]
            Xs = np.stack([codificar(a, b, c) for a, b, c, _ in sub])
            ys = np.array([d for *_, d in sub])
            acum[i].append((red.exactitud(Xs, ys), evaluar(red, sub, False)[0], evaluar(red, sub, True)[0]))
    for i, v in acum.items():
        a = np.mean(v, axis=0)
        print(f"mapa {i:<3}{a[0]:8.3f}{a[1]:12.3f}{a[2]:13.3f}")
    print("  Referencia: los 3 mapas se usan en entrenamiento; solo cambian los pares inicio-meta.")


def exp7():
    print("\n== 7. Profesor DFS vs BFS vs A* (400 épocas) ==")
    print("   'extra' = pasos dados / pasos óptimos (1.00 = ruta óptima)")
    imprimir_cabecera()
    for prof in ('DFS', 'BFS', 'A*'):
        corrida(prof, 400, profesor=prof)
    # calidad de las etiquetas: longitud de la ruta que seguiría el propio profesor
    print("  Calidad del profesor (longitud de su ruta / óptima, media sobre todos los pares):")
    for prof in ('DFS', 'BFS', 'A*'):
        r = []
        for i, mapa in enumerate(MAPAS):
            for meta in libres(mapa):
                dist = distancias_bfs(mapa, meta)
                for pos in libres(mapa):
                    if pos != meta and pos in dist:
                        if prof == 'BFS':
                            r.append(1.0)
                        else:
                            ruta = (camino_dfs if prof == 'DFS' else camino_astar)(mapa, pos, meta)
                            r.append((len(ruta) - 1) / dist[pos])
        print(f"    {prof:<4}: {np.mean(r):.3f}")


EXPERIMENTOS = {1: exp1, 2: exp2, 3: exp3, 4: exp4, 5: exp5, 6: exp6, 7: exp7}

if __name__ == '__main__':
    pedidos = [int(a) for a in sys.argv[1:]] or list(EXPERIMENTOS)
    for n in pedidos:
        EXPERIMENTOS[n]()
