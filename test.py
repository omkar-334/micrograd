from engine import MLP

n = MLP(3, [4, 4, 1])

xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, -1.0, -1.0]]

ys = [1.0, -1.0, -1.0, 1.0]

for k in range(100):
    # forward pass
    ypred = [n(x) for x in xs]
    loss = sum([(yout - ygt) * (yout - ygt) for ygt, yout in zip(ys, ypred)])

    # backward pass
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()

    # update
    # learning_rate = 1.0 - 0.9 * k / 100
    for p in n.parameters():
        p.data += -0.1 * p.grad

    print(k, loss.data)

print(ypred)
https://github.com/omkar-334/micrograd.git