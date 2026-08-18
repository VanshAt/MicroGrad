"""
examples/day06_demo.py
-----------------------
Day 6 -- Neuron / Layer / MLP: a real tiny neural net.

Run with:
    python examples/day06_demo.py
"""

import random
from micrograd import Value, Neuron, Layer, MLP

SEP = "=" * 60


def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


# -----------------------------------------------------------------
# 1. Single Neuron
# -----------------------------------------------------------------
section("1. Single Neuron")

random.seed(0)
n = Neuron(3, activation="tanh")
print(n)
print(f"  weights : {[round(w.data, 4) for w in n.weights]}")
print(f"  bias    : {n.bias.data}")

x = [2.0, -1.0, 0.5]
out = n(x)
print(f"  forward({x}) = {out.data:.6f}   (tanh, bounded in (-1,1))")

# Backward
out.backward()
print(f"  d(out)/d(w0) = {n.weights[0].grad:.6f}")
print(f"  d(out)/d(b)  = {n.bias.grad:.6f}")
print(f"  # parameters: {len(n.parameters())}")


# -----------------------------------------------------------------
# 2. Layer
# -----------------------------------------------------------------
section("2. Layer (2 inputs -> 4 neurons)")

random.seed(1)
layer = Layer(2, 4, activation="relu")
print(layer)

outs = layer([1.0, -1.0])
print(f"  outputs: {[round(o.data, 4) for o in outs]}  (relu, non-negative)")
print(f"  # parameters: {len(layer.parameters())}")


# -----------------------------------------------------------------
# 3. MLP -- architecture
# -----------------------------------------------------------------
section("3. MLP(2, [4, 4, 1]) -- architecture")

random.seed(2)
model = MLP(2, [4, 4, 1], activation="tanh")
print(model)
print(f"  Layers  : {len(model.layers)}")
print(f"  Params  : {len(model.parameters())}")
print(f"  Expected: (2*4+4) + (4*4+4) + (4*1+1) = 12+20+5 = 37")

out = model([1.0, 0.5])
print(f"  forward([1.0, 0.5]) = {out.data:.6f}")


# -----------------------------------------------------------------
# 4. Training loop -- binary classification
# -----------------------------------------------------------------
section("4. Training loop  (20 steps, MSE loss)")

random.seed(42)
model = MLP(2, [4, 1], activation="tanh")

xs = [
    [ 2.0,  3.0],
    [-1.0, -1.0],
    [ 1.0, -2.0],
    [-3.0,  1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

lr = 0.05

print(f"  {'step':>4}  {'loss':>10}")
print(f"  {'-'*4}  {'-'*10}")

for step in range(21):
    preds = [model(x) for x in xs]
    loss = sum((p - t)**2 for p, t in zip(preds, ys)) * (1.0 / len(ys))

    if step % 5 == 0:
        print(f"  {step:>4}  {loss.data:>10.6f}")

    model.zero_grad()
    loss.backward()

    for p in model.parameters():
        p.data -= lr * p.grad

print("\n  Final predictions vs targets:")
preds = [model(x) for x in xs]
for x, pred, target in zip(xs, preds, ys):
    ok = "OK" if (pred.data > 0) == (target > 0) else "WRONG"
    print(f"    [{ok}]  input={x}  pred={pred.data:+.4f}  target={target:+.1f}")


# -----------------------------------------------------------------
# 5. Linear regression -- single neuron learns y = 2x
# -----------------------------------------------------------------
section("5. Linear regression  (single neuron, y = 2x)")

random.seed(1)
model = MLP(1, [1], out_activation="linear")

xs_reg = [[float(i)] for i in range(-4, 5)]
ys_reg = [2.0 * x[0] for x in xs_reg]

lr = 0.01
for _ in range(200):
    preds = [model(x) for x in xs_reg]
    loss = sum((p - t)**2 for p, t in zip(preds, ys_reg)) * (1.0 / len(ys_reg))
    model.zero_grad()
    loss.backward()
    for p in model.parameters():
        p.data -= lr * p.grad

w = model.layers[0].neurons[0].weights[0].data
b = model.layers[0].neurons[0].bias.data
print(f"  Learned weight: {w:.4f}  (target 2.0)")
print(f"  Learned bias  : {b:.4f}  (target 0.0)")
print(f"  Final MSE     : {loss.data:.6f}")

print(f"\n{SEP}")
print("  Day 6 complete -- Neuron / Layer / MLP all working!")
print(SEP)
