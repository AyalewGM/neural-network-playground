# Neural Network Playground

Explore **how a neural network learns** through an interactive browser-based classifier, built with React, TypeScript and Vite.

## What you can explore

- Four reproducible synthetic datasets: **XOR**, circles, linearly separable points, and spirals.
- A transparent **2-input → 1–4 configurable hidden layers → sigmoid output** feedforward neural network. Each hidden layer can contain 2–16 neurons.
- Add or remove hidden layers, change each layer's width (2–16 neurons), and experiment with **tanh/ReLU** activation and learning rate. Editing the architecture resets training.
- Run, pause, reset, or advance training one full-batch gradient step at a time.
- Watch a **decision-boundary heatmap** evolve and inspect loss and classification accuracy.
- Inspect the mathematical foundations below; all computation runs locally in your browser.

The default experiment is XOR, a classic example of a problem that cannot be solved by a single linear decision boundary.

## The mathematics

A neuron performs an affine transformation and applies an activation function:

$$
z_j = \sum_{i=1}^{d} w_{ji}x_i+b_j,\qquad a_j=\phi(z_j).
$$

For classification, the output layer uses the logistic sigmoid

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

to produce a probability \(p\in (0,1)\).

For labels \(y\in\{0,1\}\), the binary cross-entropy objective is

$$
\mathcal L(\theta)=-\frac{1}{N}\sum_{i=1}^{N}\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right].
$$

**Backpropagation** applies the chain rule to compute derivatives through the layers. Gradient descent then updates each weight and bias:

$$
\theta_{t+1}=\theta_t-\eta\nabla_\theta\mathcal L(\theta_t).
$$

In the implementation, the output-layer error simplifies to \(\delta^{(L)}=p-y\) for sigmoid plus binary cross-entropy; hidden-layer errors are propagated through the weight matrices and multiplied by the activation derivatives. Weight gradients are averaged over the full synthetic dataset.

### Why this remains important

Neural networks connect **linear algebra, multivariable calculus, optimization and statistics** to modern machine learning. Seeing the complete training loop demystifies large-scale AI models: prediction, measuring error, differentiating that error and updating parameters. Experiments also expose why model capacity, initialization, activation choice and learning rate matter.

These demonstrations are intentionally educational. A small network will not always solve a spiral dataset with the default configuration, and a smooth-looking decision boundary does not guarantee generalization. The displayed accuracy uses the same sample set used for training.

### Experiments to try

1. Start with XOR and try only two hidden neurons. Increase to eight. How does the boundary change?
2. Compare circles under tanh and ReLU.
3. Increase learning rate until optimization becomes unstable.
4. Try spirals. How much hidden-layer capacity is needed to fit the pattern?
5. Reset after training to see the effect of deterministic initialization.

## Development

Requires Node.js 20.19+ or Node.js 22.12+.

```bash
npm install
npm test
npm run dev
npm run build
```

The Vite base path is configured for GitHub Pages deployment at `/neural-network-playground/`. In **Settings → Pages**, select **GitHub Actions** as the publishing source. Deployments run after a successful merge into `main`.

## Architecture

- `src/nn.ts` — seeded datasets, neural-network initialization, forward propagation, cross-entropy and analytic backpropagation.
- `src/App.tsx` — training controls and canvas-based classifier visualization.
- `src/nn.test.ts` — deterministic initialization, probability, training improvement and mutation-safety checks.

**Current limitations:** Up to four hidden layers; full-batch gradient descent only; CPU-only browser training; no independent validation dataset, regularization, momentum/Adam, or Web Worker yet. These are suitable next milestones.

## Recommended reading

- [Michael Nielsen, Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
- [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)
- [CS231n: Backpropagation, Intuitions](https://cs231n.github.io/optimization-2/)

Independent educational application in the GLife family of mathematical projects.

## Interactive architecture builder

Use **Add layer** to append another hidden layer, **Remove** to delete a selected hidden layer, and each layer's slider to change its neuron count. The architecture preview shows `2 → h₁ → h₂ → … → 1`. Layers are initialized afresh after every architecture change, preventing mismatched weight dimensions or stale training history. Backpropagation traverses every configured hidden layer.

Try the XOR dataset using `2 → 2 → 1`, then compare with `2 → 8 → 4 → 1`. Bigger networks are not guaranteed to learn faster or generalize better.
