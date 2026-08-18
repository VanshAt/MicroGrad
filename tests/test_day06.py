import math
import random

import pytest

from micrograd import MLP, Layer, Neuron, Value


@pytest.fixture(autouse=True)
def seed():
    random.seed(0)


class TestNeuronConstruction:
    def test_default_activation_is_tanh(self):
        assert Neuron(3).activation == "tanh"

    def test_weight_count_matches_n_in(self):
        for n_in in (1, 3, 10):
            assert len(Neuron(n_in).weights) == n_in

    def test_bias_starts_at_zero(self):
        assert Neuron(4).bias.data == 0.0

    def test_weights_are_value_nodes(self):
        assert all(isinstance(w, Value) for w in Neuron(3).weights)

    def test_bias_is_value_node(self):
        assert isinstance(Neuron(2).bias, Value)

    def test_weights_within_kaiming_bounds(self):
        n_in = 100
        limit = 1.0 / math.sqrt(n_in)
        for w in Neuron(n_in).weights:
            assert -limit <= w.data <= limit

    def test_all_activations_accepted(self):
        for act in ("tanh", "relu", "sigmoid", "linear"):
            Neuron(2, activation=act)

    def test_invalid_activation_raises(self):
        with pytest.raises(ValueError, match="activation must be one of"):
            Neuron(2, activation="gelu")

    def test_zero_n_in_raises(self):
        with pytest.raises(ValueError, match="n_in must be a positive integer"):
            Neuron(0)

    def test_negative_n_in_raises(self):
        with pytest.raises(ValueError, match="n_in must be a positive integer"):
            Neuron(-1)

    def test_repr_contains_key_info(self):
        r = repr(Neuron(3, activation="relu"))
        assert "n_in=3" in r and "relu" in r


class TestNeuronParameters:
    def test_parameter_count(self):
        assert len(Neuron(4).parameters()) == 5

    def test_parameters_are_all_values(self):
        assert all(isinstance(p, Value) for p in Neuron(3).parameters())

    def test_bias_is_last_parameter(self):
        n = Neuron(3)
        assert n.parameters()[-1] is n.bias


class TestNeuronForward:
    def test_output_is_value(self):
        assert isinstance(Neuron(2)([1.0, 2.0]), Value)

    def test_tanh_output_bounded(self):
        out = Neuron(3, activation="tanh")([1.0, -1.0, 0.5])
        assert -1.0 <= out.data <= 1.0

    def test_relu_output_non_negative(self):
        n = Neuron(2, activation="relu")
        for _ in range(20):
            out = n([random.uniform(-5, 5), random.uniform(-5, 5)])
            assert out.data >= 0.0

    def test_sigmoid_output_in_unit_interval(self):
        n = Neuron(2, activation="sigmoid")
        for _ in range(20):
            out = n([random.uniform(-5, 5), random.uniform(-5, 5)])
            assert 0.0 < out.data < 1.0

    def test_linear_output_is_raw_dot_plus_bias(self):
        n = Neuron(2, activation="linear")
        n.weights[0].data = 1.0
        n.weights[1].data = 2.0
        n.bias.data = 0.5
        out = n([3.0, 4.0])
        assert abs(out.data - (1.0 * 3.0 + 2.0 * 4.0 + 0.5)) < 1e-10

    def test_accepts_value_inputs(self):
        assert isinstance(Neuron(2)([Value(1.0), Value(2.0)]), Value)

    def test_accepts_mixed_inputs(self):
        assert isinstance(Neuron(2)([Value(1.0), 2.0]), Value)

    def test_wrong_input_length_raises(self):
        with pytest.raises(ValueError, match="Expected 3 inputs"):
            Neuron(3)([1.0, 2.0])


class TestNeuronGradients:
    def test_backward_populates_weight_grads(self):
        n = Neuron(2, activation="linear")
        out = n([1.0, 2.0])
        out.backward()
        for w in n.weights:
            assert w.grad != 0.0

    def test_backward_populates_bias_grad(self):
        n = Neuron(2, activation="linear")
        out = n([1.0, 2.0])
        out.backward()
        assert n.bias.grad != 0.0

    def test_tanh_gradient_correct(self):
        n = Neuron(1, activation="tanh")
        n.weights[0].data = 0.5
        n.bias.data = 0.0
        x_val = 2.0
        out = n([x_val])
        out.backward()
        t = math.tanh(0.5 * x_val)
        expected = (1 - t**2) * x_val
        assert abs(n.weights[0].grad - expected) < 1e-9

    def test_relu_zero_grad_for_dead_neuron(self):
        n = Neuron(2, activation="relu")
        n.weights[0].data = -10.0
        n.weights[1].data = -10.0
        n.bias.data = 0.0
        out = n([1.0, 1.0])
        out.backward()
        assert n.weights[0].grad == 0.0
        assert n.weights[1].grad == 0.0


class TestLayerConstruction:
    def test_neuron_count(self):
        assert len(Layer(3, 5).neurons) == 5

    def test_all_neurons_have_correct_n_in(self):
        for neuron in Layer(4, 3).neurons:
            assert len(neuron.weights) == 4

    def test_default_activation_is_tanh(self):
        assert all(n.activation == "tanh" for n in Layer(2, 3).neurons)

    def test_activation_propagated(self):
        assert all(n.activation == "relu" for n in Layer(2, 3, activation="relu").neurons)

    def test_repr_contains_key_info(self):
        r = repr(Layer(3, 4, activation="sigmoid"))
        assert "n_in=3" in r and "n_out=4" in r and "sigmoid" in r


class TestLayerParameters:
    def test_parameter_count(self):
        assert len(Layer(3, 4).parameters()) == 4 * (3 + 1)

    def test_all_parameters_are_values(self):
        assert all(isinstance(p, Value) for p in Layer(2, 3).parameters())


class TestLayerForward:
    def test_output_length_equals_n_out(self):
        assert len(Layer(3, 5)([1.0, 2.0, 3.0])) == 5

    def test_output_elements_are_values(self):
        assert all(isinstance(o, Value) for o in Layer(2, 3)([0.5, -0.5]))

    def test_single_neuron_layer_still_returns_list(self):
        outs = Layer(2, 1)([1.0, 2.0])
        assert isinstance(outs, list) and len(outs) == 1


class TestMLPConstruction:
    def test_layer_count(self):
        assert len(MLP(2, [4, 4, 1]).layers) == 3

    def test_single_layer_mlp(self):
        assert len(MLP(3, [1]).layers) == 1

    def test_layer_widths(self):
        assert [len(l.neurons) for l in MLP(2, [4, 3, 1]).layers] == [4, 3, 1]

    def test_hidden_layers_use_relu(self):
        model = MLP(2, [4, 4, 1], activation="relu")
        assert model.layers[0].neurons[0].activation == "relu"
        assert model.layers[1].neurons[0].activation == "relu"

    def test_output_layer_linear_by_default(self):
        assert MLP(2, [4, 1]).layers[-1].neurons[0].activation == "linear"

    def test_out_activation_respected(self):
        assert MLP(2, [4, 1], out_activation="sigmoid").layers[-1].neurons[0].activation == "sigmoid"

    def test_empty_layer_sizes_raises(self):
        with pytest.raises(ValueError, match="layer_sizes must contain"):
            MLP(2, [])

    def test_repr_contains_layers(self):
        assert "Layer" in repr(MLP(2, [3, 1]))


class TestMLPParameters:
    def test_parameter_count_2_4_4_1(self):
        # Layer(2->4):4*(2+1)=12, Layer(4->4):4*(4+1)=20, Layer(4->1):1*(4+1)=5 -> 37
        assert len(MLP(2, [4, 4, 1]).parameters()) == 37

    def test_parameter_count_single_layer(self):
        assert len(MLP(3, [1]).parameters()) == 4

    def test_all_parameters_are_values(self):
        assert all(isinstance(p, Value) for p in MLP(2, [3, 1]).parameters())

    def test_no_duplicate_parameters(self):
        params = MLP(2, [4, 1]).parameters()
        assert len(params) == len(set(id(p) for p in params))


class TestMLPForward:
    def test_single_output_returns_value(self):
        assert isinstance(MLP(2, [4, 1])([1.0, 2.0]), Value)

    def test_multi_output_returns_list(self):
        out = MLP(2, [4, 3])([1.0, 2.0])
        assert isinstance(out, list) and len(out) == 3

    def test_accepts_value_inputs(self):
        assert isinstance(MLP(2, [3, 1])([Value(1.0), Value(2.0)]), Value)

    def test_deep_mlp_runs_without_error(self):
        assert isinstance(MLP(4, [8, 8, 8, 1])([0.1, 0.2, 0.3, 0.4]), Value)

    def test_output_is_finite(self):
        assert MLP(2, [4, 4, 1])([1.0, -1.0]).is_finite()


class TestMLPGradients:
    def test_zero_grad_resets_all(self):
        model = MLP(2, [3, 1])
        model([1.0, 2.0]).backward()
        model.zero_grad()
        assert all(p.grad == 0.0 for p in model.parameters())

    def test_gradient_does_not_accumulate_across_steps(self):
        model = MLP(2, [3, 1])
        x = [1.0, 2.0]
        model(x).backward()
        grads1 = [p.grad for p in model.parameters()]
        model.zero_grad()
        model(x).backward()
        grads2 = [p.grad for p in model.parameters()]
        for g1, g2 in zip(grads1, grads2):
            assert abs(g1 - g2) < 1e-10


class TestTrainingLoop:
    def _mse(self, model, xs, ys):
        preds = [model(x) for x in xs]
        return sum((p - t)**2 for p, t in zip(preds, ys)) * (1.0 / len(ys))

    def test_loss_decreases_tanh(self):
        random.seed(42)
        model = MLP(2, [4, 1], activation="tanh")
        xs = [[2.0, 3.0], [-1.0, -1.0], [1.0, -2.0], [-3.0, 1.0]]
        ys = [1.0, -1.0, -1.0, 1.0]
        lr = 0.05
        initial = self._mse(model, xs, ys).data
        for _ in range(20):
            loss = self._mse(model, xs, ys)
            model.zero_grad()
            loss.backward()
            for p in model.parameters():
                p.data -= lr * p.grad
        assert self._mse(model, xs, ys).data < initial

    def test_loss_decreases_relu(self):
        random.seed(7)
        model = MLP(2, [8, 1], activation="relu")
        xs = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
        ys = [1.0, 1.0, -1.0, -1.0]
        lr = 0.01
        initial = self._mse(model, xs, ys).data
        for _ in range(30):
            loss = self._mse(model, xs, ys)
            model.zero_grad()
            loss.backward()
            for p in model.parameters():
                p.data -= lr * p.grad
        assert self._mse(model, xs, ys).data < initial

    def test_single_neuron_fits_linear_target(self):
        random.seed(1)
        model = MLP(1, [1], out_activation="linear")
        xs = [[float(i)] for i in range(-4, 5)]
        ys = [2.0 * x[0] for x in xs]
        lr = 0.01
        for _ in range(200):
            loss = self._mse(model, xs, ys)
            model.zero_grad()
            loss.backward()
            for p in model.parameters():
                p.data -= lr * p.grad
        w = model.layers[0].neurons[0].weights[0].data
        b = model.layers[0].neurons[0].bias.data
        assert abs(w - 2.0) < 0.05, f"w={w:.4f}"
        assert abs(b) < 0.05, f"b={b:.4f}"


class TestZeroGrad:
    def test_mlp_zero_grad_clears_params(self):
        model = MLP(2, [3, 1])
        model([1.0, 2.0]).backward()
        model.zero_grad()
        assert all(p.grad == 0.0 for p in model.parameters())

    def test_mlp_zero_grad_leaves_data_intact(self):
        model = MLP(2, [3, 1])
        data_before = [p.data for p in model.parameters()]
        model([1.0, 2.0]).backward()
        model.zero_grad()
        for d1, d2 in zip(data_before, [p.data for p in model.parameters()]):
            assert d1 == d2
