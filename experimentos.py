"""Experimentos con la red de Neurito (misma red y entrenamiento que main.py, sin pygame)."""

import numpy as np

actions = ("COMER", "DORMIR", "JUGAR")


def policy(data):
    return np.argmax(np.column_stack((data[:, 0], 1 - data[:, 1], data[:, 2])), axis=1)


def relu(values):
    return np.maximum(0, values)


def softmax(values):
    exp_values = np.exp(values - np.max(values, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def train(hidden_size=6, epochs=1200, lr=0.1, seed=42, log_every=None):
    """Entrena igual que main.py. Con los valores por defecto reproduce main.py exactamente."""
    np.random.seed(seed)
    X = np.random.rand(3000, 3)
    targets = policy(X)
    W1 = np.random.randn(3, hidden_size) * 0.5
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, 3) * 0.5
    b2 = np.zeros((1, 3))
    history = []
    count = len(X)
    for epoch in range(epochs):
        z1 = X @ W1 + b1
        hidden = relu(z1)
        probabilities = softmax(hidden @ W2 + b2)
        loss = -np.mean(np.log(probabilities[np.arange(count), targets] + 1e-10))
        history.append(loss)
        if log_every and epoch % log_every == 0:
            print(f"  época {epoch:5d} | pérdida {loss:.4f}")
        delta2 = probabilities.copy()
        delta2[np.arange(count), targets] -= 1
        delta2 /= count
        grad_W2 = hidden.T @ delta2
        grad_b2 = np.sum(delta2, axis=0, keepdims=True)
        delta1 = (delta2 @ W2.T) * (z1 > 0)
        grad_W1 = X.T @ delta1
        grad_b1 = np.sum(delta1, axis=0, keepdims=True)
        W1 -= lr * grad_W1
        b1 -= lr * grad_b1
        W2 -= lr * grad_W2
        b2 -= lr * grad_b2
    test = np.random.rand(1000, 3)
    params = (W1, b1, W2, b2)
    accuracy = np.mean(np.argmax(predict(params, test), axis=1) == policy(test))
    return params, history, accuracy, test


def predict(params, data):
    W1, b1, W2, b2 = params
    return softmax(relu(data @ W1 + b1) @ W2 + b2)


def fmt(p):
    return " | ".join(f"{a} {v * 100:5.1f}%" for a, v in zip(actions, p))


def sweep(params, fixed, index, name):
    print(f"\n### Variando {name} (fijos: hambre={fixed[0]}, energía={fixed[1]}, aburrimiento={fixed[2]})")
    print("| valor | cansancio | predicción red | política | probabilidades |")
    print("|---|---|---|---|---|")
    for v in np.round(np.arange(0, 1.01, 0.1), 2):
        x = np.array([fixed], dtype=float)
        x[0, index] = v
        p = predict(params, x)[0]
        print(f"| {v:.1f} | {1 - x[0, 1]:.1f} | {actions[np.argmax(p)]} | {actions[policy(x)[0]]} | {fmt(p)} |")


def find_switch(params, fixed, index):
    values = np.linspace(0, 1, 1001)
    x = np.tile(np.array(fixed, dtype=float), (len(values), 1))
    x[:, index] = values
    net = np.argmax(predict(params, x), axis=1)
    pol = policy(x)
    net_sw = [(values[i], actions[net[i - 1]], actions[net[i]]) for i in range(1, len(values)) if net[i] != net[i - 1]]
    pol_sw = [(values[i], actions[pol[i - 1]], actions[pol[i]]) for i in range(1, len(values)) if pol[i] != pol[i - 1]]
    return net_sw, pol_sw


if __name__ == "__main__":
    print("=== Línea base (igual que main.py) ===")
    base, hist, acc, test = train(log_every=200)
    print(f"Pérdida final: {hist[-1]:.4f} | Precisión test: {acc:.2%}")

    print("\n=== Experimento 1: dos fijas, una variable ===")
    cases = [([0.3, 0.6, 0.0], 2, "aburrimiento"), ([0.0, 0.6, 0.3], 0, "hambre"), ([0.3, 0.0, 0.3], 1, "energía")]
    for fixed, index, name in cases:
        sweep(base, fixed, index, name)
        net_sw, pol_sw = find_switch(base, fixed, index)
        print(f"Cambios red: {[(round(v, 3), a, b) for v, a, b in net_sw]}")
        print(f"Cambios política: {[(round(v, 3), a, b) for v, a, b in pol_sw]}")

    print("\n=== Experimento 2: prioridades casi iguales ===")
    print("| hambre | energía (cansancio) | aburrimiento | política | red | probabilidades |")
    print("|---|---|---|---|---|---|")
    near = [
        [0.50, 0.50, 0.50], [0.51, 0.50, 0.50], [0.50, 0.49, 0.50], [0.50, 0.50, 0.51],
        [0.80, 0.20, 0.80], [0.81, 0.20, 0.79], [0.20, 0.80, 0.20], [0.90, 0.90, 0.10],
        [0.90, 0.10, 0.10],
    ]
    for n in near:
        x = np.array([n])
        p = predict(base, x)[0]
        print(f"| {n[0]:.2f} | {n[1]:.2f} ({1 - n[1]:.2f}) | {n[2]:.2f} | {actions[policy(x)[0]]} | {actions[np.argmax(p)]} | {fmt(p)} |")

    print("\n=== Experimento 3: tamaño de la capa oculta ===")
    print("| neuronas | pérdida final | precisión test |")
    print("|---|---|---|")
    for h in (3, 6, 10, 20):
        _, hist_h, acc_h, _ = train(hidden_size=h)
        print(f"| {h} | {hist_h[-1]:.4f} | {acc_h:.2%} |")
    print("\nPromedio en 5 semillas:")
    print("| neuronas | pérdida media | precisión media | precisión mín-máx |")
    print("|---|---|---|---|")
    for h in (3, 6, 10, 20):
        r = [train(hidden_size=h, seed=s) for s in range(5)]
        losses = [x[1][-1] for x in r]
        accs = [x[2] for x in r]
        print(f"| {h} | {np.mean(losses):.4f} | {np.mean(accs):.2%} | {min(accs):.2%} - {max(accs):.2%} |")

    print("\n=== Experimento 4a: épocas (lr=0.1) ===")
    print("| épocas | pérdida final | precisión test |")
    print("|---|---|---|")
    for e in (100, 300, 1200, 5000, 20000):
        _, hist_e, acc_e, _ = train(epochs=e)
        print(f"| {e} | {hist_e[-1]:.4f} | {acc_e:.2%} |")

    print("\n=== Experimento 4b: tasa de aprendizaje (1200 épocas) ===")
    print("| lr | pérdida ép. 0 | ép. 300 | ép. 600 | ép. 1199 | mín | máx tras ép. 100 | precisión |")
    print("|---|---|---|---|---|---|---|---|")
    for lr in (0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 20.0):
        _, h, a, _ = train(lr=lr)
        h = np.array(h)
        print(f"| {lr} | {h[0]:.4f} | {h[300]:.4f} | {h[600]:.4f} | {h[-1]:.4f} | {h.min():.4f} | {h[100:].max():.4f} | {a:.2%} |")

    print("\n=== Explicación: precisión engañosa ===")
    pol = policy(test)
    pred = np.argmax(predict(base, test), axis=1)
    scores = np.sort(np.column_stack((test[:, 0], 1 - test[:, 1], test[:, 2])), axis=1)
    margin = scores[:, 2] - scores[:, 1]
    for lo, hi in ((0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 1.01)):
        m = (margin >= lo) & (margin < hi)
        print(f"Margen [{lo}, {hi}): {m.sum():4d} ejemplos, precisión {np.mean(pred[m] == pol[m]):.2%}")
    print("Matriz de confusión (filas=política, columnas=red):")
    for i, a in enumerate(actions):
        print(f"  {a:7s}", [int(np.sum((pol == i) & (pred == j))) for j in range(3)])
    print("Distribución de clases en test:", {a: int(np.sum(pol == i)) for i, a in enumerate(actions)})
    print(f"Precisión de un modelo que siempre dice la clase mayoritaria: {np.max(np.bincount(pol)) / len(pol):.2%}")
    conf = np.max(predict(base, test), axis=1)
    print(f"Confianza media en aciertos: {conf[pred == pol].mean():.2%} | en errores: {conf[pred != pol].mean():.2%}")
    ood = np.array([[2.0, 0.5, 0.5], [-1.0, 0.5, 0.5], [0.5, 0.5, 3.0]])
    for x in ood:
        print(f"Fuera de rango {x}: red={actions[np.argmax(predict(base, x[None]))]} {fmt(predict(base, x[None])[0])}")

    W1, b1, W2, b2 = base
    print("\n=== Pesos y sesgos aprendidos ===")
    print("W1 (3x6):\n", np.round(W1, 3))
    print("b1:", np.round(b1, 3))
    print("W2 (6x3):\n", np.round(W2, 3))
    print("b2:", np.round(b2, 3))
    print("Salida con entrada [0,0,0] (solo sesgos actúan):", fmt(predict(base, np.zeros((1, 3)))[0]))
