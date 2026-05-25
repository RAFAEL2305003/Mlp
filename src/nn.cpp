#include <cmath>
#include "../include/linear_layer.h"
#include "../include/activation_layer.h"
#include "../include/conv1d_layer.h"
#include "../include/pooling_layer.h"
#include "../include/loss_layer.h"
#include "../include/rapidcsv.h"

namespace nn {

	class counter
	{
		public:
			size_t idx;

			counter() : idx(0) {}
	};

	// Slow path: only used for one-shot full-dataset predictions and accuracy.
	// Per-batch hot path uses copy_rows_gpu below instead.
	math::matrix<float> get_cols(const math::matrix<float>& m, size_t low, size_t high)
	{
		auto [rows, cols] = m.shape();
		assert(low < high);
		assert(low <= cols && high <= cols);

		std::size_t width = high - low;
		math::matrix<float> r(rows, width);
		const float* src = m.elements;
		float* dst = r.elements;
		std::size_t mc = cols;
		std::size_t lo = low;
		math::q.parallel_for(sycl::range<2>(rows, width), [=](sycl::id<2> idx) {
			std::size_t i = idx[0];
			std::size_t j = idx[1];
			dst[i * width + j] = src[i * mc + (lo + j)];
		});
		return r;
	}

	// Copy `batch_size` contiguous rows from `src` (row-major, src_cols wide) starting at
	// row offset `row_offset` into `dst` (must already have shape (batch_size, src_cols)).
	// Single async device memcpy — replaces the old CPU nested-loop that thrashed
	// CUDA managed-memory pages every batch.
	void copy_rows_gpu(const math::matrix<float>& src,
					   math::matrix<float>& dst,
					   std::size_t row_offset,
					   std::size_t batch_size)
	{
		auto [src_rows, src_cols] = src.shape();
		assert(row_offset + batch_size <= src_rows);
		assert(dst.shape().first == batch_size && dst.shape().second == src_cols);
		math::q.memcpy(dst.elements,
					   src.elements + row_offset * src_cols,
					   batch_size * src_cols * sizeof(float));
	}

class dense_layer
{
	private:
	size_t epochs;

	float learning_rate;

	size_t num_layers;

	size_t input_size;
	size_t output_size;

	math::matrix<float> dataset;
	math::matrix<float> X_full;
	math::matrix<float> Y_full;
	math::matrix<float> x_batch_buf;
	math::matrix<float> y_batch_buf;

	std::vector<nn::conv1d_layer> conv;
	std::vector<nn::activation_layer> conv_activation;
	std::vector<nn::pooling_layer> pool;
	std::vector<nn::linear_layer> linear;
	std::vector<nn::activation_layer> activation;
	nn::loss_layer loss;

	public:
	dense_layer(size_t epochs,
				float learning_rate,
				std::vector<size_t> layers,
				std::vector<activation::type> activations,
				loss::type loss_type,
				const math::matrix<float>& dataset)
		:
			epochs(epochs),

			learning_rate(learning_rate),

			num_layers(layers.size() - 1), // excluding input layer

			input_size(layers[0]),

			output_size(layers[layers.size() - 1]),

			dataset(dataset)
		{
			assert(layers.size() > 1);
			assert(activations.size() == num_layers);

			math::random<float> rng;

			conv.resize(0);
			conv_activation.resize(0);
			pool.resize(0);
			linear.resize(num_layers);
			activation.resize(num_layers);

			for(size_t i = 0; i < num_layers; i++)
			{
				linear[i] = nn::linear_layer(layers[i], layers[i + 1], rng);
				activation[i] = nn::activation_layer(activations[i]);
			}

			loss = nn::loss_layer(loss_type);

			X_full = nn::get_cols(this->dataset, 0, this->input_size);
			Y_full = nn::get_cols(this->dataset, this->input_size, this->input_size + this->output_size);
		}

	dense_layer(size_t epochs,
				float learning_rate,
				size_t input_size,
				std::vector<conv1d::config> conv_configs,
				std::vector<activation::type> conv_activations,
				std::vector<pooling::config> pool_configs,
				std::vector<size_t> layers,
				std::vector<activation::type> activations,
				loss::type loss_type,
				const math::matrix<float>& dataset)
		:
			epochs(epochs),

			learning_rate(learning_rate),

			num_layers(layers.size() - 1),

			input_size(input_size),

			output_size(layers[layers.size() - 1]),

			dataset(dataset)
		{
			assert(conv_configs.size() == conv_activations.size());
			assert(conv_configs.size() == pool_configs.size());
			assert(layers.size() > 1);
			assert(activations.size() == num_layers);

			math::random<float> rng;

			size_t current_input_size = this->input_size;

			conv.resize(conv_configs.size());
			conv_activation.resize(conv_configs.size());
			pool.resize(pool_configs.size());

			for(size_t i = 0; i < conv_configs.size(); i++)
			{
				conv[i] = nn::conv1d_layer(current_input_size,
										   conv_configs[i].filters,
										   conv_configs[i].kernel_size,
										   rng,
										   conv_configs[i].conv_type,
										   conv_configs[i].stride);
				conv_activation[i] = nn::activation_layer(conv_activations[i]);

				pool[i] = nn::pooling_layer(pool_configs[i].pool_type,
											conv[i].output_size,
											conv_configs[i].filters,
											pool_configs[i].pool_size,
											pool_configs[i].stride);

				current_input_size = conv_configs[i].filters * pool[i].output_size;
			}

			assert(layers[0] == current_input_size);

			linear.resize(num_layers);
			activation.resize(num_layers);

			for(size_t i = 0; i < num_layers; i++)
			{
				linear[i] = nn::linear_layer(layers[i], layers[i + 1], rng);
				activation[i] = nn::activation_layer(activations[i]);
			}

			loss = nn::loss_layer(loss_type);

			X_full = nn::get_cols(this->dataset, 0, this->input_size);
			Y_full = nn::get_cols(this->dataset, this->input_size, this->input_size + this->output_size);
		}

	math::matrix<float> feedforward(const math::matrix<float>& x_batch)
	{
		math::matrix<float> features = x_batch;
		for(size_t i = 0; i < conv.size(); i++)
		{
			features = conv[i].forward(features);
			features = conv_activation[i].forward(features);
			features = pool[i].forward(features);
		}

		math::matrix fwd = linear[0].forward(features);
		math::matrix act = activation[0].forward(fwd);
		for(size_t i = 1; i < num_layers; i++)
		{
			fwd = linear[i].forward(act);
			act = activation[i].forward(fwd);
		}

		return act;
	}

	void backward(const math::matrix<float>& Y, const math::matrix<float>& y_batch)
	{
		math::matrix dL_dy = loss.backward(Y, y_batch);
		math::matrix delta = activation[num_layers - 1].backward(dL_dy);
		math::matrix delta_last = linear[num_layers - 1].backward(delta);
		for(int i = num_layers - 2; i >= 0; i--)
		{
			delta = activation[i].backward(delta_last);
			delta_last = linear[i].backward(delta);
		}

		for(size_t i = conv.size(); i-- > 0;)
		{
			delta_last = pool[i].backward(delta_last);
			delta_last = conv_activation[i].backward(delta_last);
			delta_last = conv[i].backward(delta_last);
		}
	}

	void update()
	{
		for(size_t i = 0; i < conv.size(); i++)
		{
			conv[i].update(learning_rate);
		}

		for(size_t i = 0; i < num_layers; i++)
		{
			linear[i].update(learning_rate);
		}
	}

	void train(size_t batch_size, counter& c)
	{
		auto [rows, cols] = dataset.shape();
		size_t num_batches = rows / batch_size;
		size_t old_batch_size = batch_size;

		size_t rmd = rows % batch_size;
		(rmd != 0) ? num_batches += 1 : num_batches;

		// Preallocate batch staging buffers on the device. We resize only for
		// the trailing partial batch — the assignment-operator fast path keeps
		// the same allocation when shapes match between iterations.
		x_batch_buf = math::matrix<float>(batch_size, input_size);
		y_batch_buf = math::matrix<float>(batch_size, output_size);

		float total_loss = 0.0;
		for(size_t e = 0; e < epochs; e++)
		{
			total_loss = 0;
			c.idx = 0;
			for(size_t i = 0; i < num_batches; i++)
			{
				if(i == num_batches - 1 && rmd != 0)
				{
					batch_size = rmd;
					x_batch_buf.ensure_shape(batch_size, input_size);
					y_batch_buf.ensure_shape(batch_size, output_size);
				}

				nn::copy_rows_gpu(X_full, x_batch_buf, c.idx, batch_size);
				nn::copy_rows_gpu(Y_full, y_batch_buf, c.idx, batch_size);
				c.idx += batch_size;

				math::matrix fwd_pass = feedforward(x_batch_buf);
				total_loss += loss.forward(fwd_pass, y_batch_buf);
				backward(fwd_pass, y_batch_buf);
				update();
				// Release transient buffers queued via deferred_frees. Without
				// this, USM allocations accumulate every batch and exhaust VRAM
				// (caused a GPU hang / display logout previously).
				math::drain();
			}
			if(e % 5 == 0)
				std::cout << "epoch: " << e << ", loss: " << total_loss / num_batches << "\n";

			batch_size = old_batch_size;
			x_batch_buf.ensure_shape(batch_size, input_size);
			y_batch_buf.ensure_shape(batch_size, output_size);
		}
		std::cout << "epoch: " << epochs << ", loss: " << total_loss / num_batches << "\n";
	}

	float accuracy(const math::matrix<float>& predicted)
	{
		auto [rows, cols] = predicted.shape();

		size_t correct = 0;
		if(cols == 1)
		{
			std::vector<float> pred_host(rows);
			std::vector<float> y_host(rows);
			math::q.memcpy(pred_host.data(), predicted.elements, rows * sizeof(float));
			math::q.memcpy(y_host.data(), Y_full.elements, rows * sizeof(float));
			math::q.wait();
			for(size_t i = 0; i < rows; i++)
			{
				float predicted_class = pred_host[i] >= 0.5f ? 1.0f : 0.0f;
				if(predicted_class == y_host[i])
				{
					correct++;
				}
			}
		}
		else
		{
			std::vector predicted_class = activation::argmax(predicted);
			std::vector actual_class = activation::argmax(Y_full);
			for(size_t i = 0; i < rows; i++)
			{
				if(predicted_class[i] == actual_class[i])
				{
					correct++;
				}
			}
		}

		return static_cast<float> (correct) / rows;
	}

	// Returns {tp, fp, fn} per class. For binary (cols==1), index 0 is the
	// positive class (label == 1). For multi-class, index c corresponds to class c.
	std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<size_t>>
	confusion_counts(const math::matrix<float>& predicted)
	{
		auto [rows, cols] = predicted.shape();
		size_t num_classes = cols == 1 ? 1 : cols;
		std::vector<size_t> tp(num_classes, 0);
		std::vector<size_t> fp(num_classes, 0);
		std::vector<size_t> fn(num_classes, 0);

		if(cols == 1)
		{
			std::vector<float> pred_host(rows);
			std::vector<float> y_host(rows);
			math::q.memcpy(pred_host.data(), predicted.elements, rows * sizeof(float));
			math::q.memcpy(y_host.data(), Y_full.elements, rows * sizeof(float));
			math::q.wait();
			for(size_t i = 0; i < rows; i++)
			{
				bool p = pred_host[i] >= 0.5f;
				bool a = y_host[i] >= 0.5f;
				if(p && a) tp[0]++;
				else if(p && !a) fp[0]++;
				else if(!p && a) fn[0]++;
			}
		}
		else
		{
			std::vector predicted_class = activation::argmax(predicted);
			std::vector actual_class = activation::argmax(Y_full);
			for(size_t i = 0; i < rows; i++)
			{
				size_t p = predicted_class[i];
				size_t a = actual_class[i];
				if(p == a)
				{
					tp[p]++;
				}
				else
				{
					fp[p]++;
					fn[a]++;
				}
			}
		}

		return {tp, fp, fn};
	}

	// Macro-averaged precision: mean of per-class TP / (TP + FP).
	// Classes with no positive predictions contribute 0 to the average.
	float precision(const math::matrix<float>& predicted)
	{
		auto [tp, fp, fn] = confusion_counts(predicted);
		size_t num_classes = tp.size();
		float sum = 0.0f;
		for(size_t c = 0; c < num_classes; c++)
		{
			size_t denom = tp[c] + fp[c];
			if(denom > 0)
			{
				sum += static_cast<float>(tp[c]) / static_cast<float>(denom);
			}
		}
		return sum / static_cast<float>(num_classes);
	}

	// Macro-averaged recall: mean of per-class TP / (TP + FN).
	float recall(const math::matrix<float>& predicted)
	{
		auto [tp, fp, fn] = confusion_counts(predicted);
		size_t num_classes = tp.size();
		float sum = 0.0f;
		for(size_t c = 0; c < num_classes; c++)
		{
			size_t denom = tp[c] + fn[c];
			if(denom > 0)
			{
				sum += static_cast<float>(tp[c]) / static_cast<float>(denom);
			}
		}
		return sum / static_cast<float>(num_classes);
	}

	// Macro-averaged F1: mean of per-class 2*P*R / (P+R).
	float f1_score(const math::matrix<float>& predicted)
	{
		auto [tp, fp, fn] = confusion_counts(predicted);
		size_t num_classes = tp.size();
		float sum = 0.0f;
		for(size_t c = 0; c < num_classes; c++)
		{
			float p_denom = static_cast<float>(tp[c] + fp[c]);
			float r_denom = static_cast<float>(tp[c] + fn[c]);
			float p = p_denom > 0.0f ? static_cast<float>(tp[c]) / p_denom : 0.0f;
			float r = r_denom > 0.0f ? static_cast<float>(tp[c]) / r_denom : 0.0f;
			if(p + r > 0.0f)
			{
				sum += 2.0f * p * r / (p + r);
			}
		}
		return sum / static_cast<float>(num_classes);
	}

};

	math::matrix<float> round(math::matrix<float>& predictions)
	{
		auto [rows, cols] = predictions.shape();
		math::matrix<float> r(rows, cols);

		for(size_t i = 0; i < rows; i++)
		{
			for(size_t j = 0; j < cols; j++)
			{
				r(i, j) = std::round(predictions(i, j));
			}
		}

		return r;
	}

};

math::matrix<float> read_csv(std::string filename)
{
    rapidcsv::Document attr_dataset(filename);
    std::vector<std::string> attrs = attr_dataset.GetColumnNames();
    assert(!attrs.empty());

    std::vector<float> attr = attr_dataset.GetColumn<float>(attrs[0]);
    math::matrix<float> dataset(attr.size(), 1, attr);
    for(size_t i = 1; i < attrs.size(); i++)
    {
        attr = attr_dataset.GetColumn<float>(attrs[i]);
        dataset = math::concat(dataset, attr);
    }

    return dataset;
}

// Z-score normalization per feature column (leaves label columns untouched).
// Features with near-zero variance (constant columns) are left as-is.
// Runs entirely on the device — data is malloc_device USM and not host-readable.
void normalize_features(math::matrix<float>& data, size_t num_features)
{
	auto [rows, cols] = data.shape();
	assert(num_features <= cols);
	if (rows == 0 || num_features == 0) return;

	float* mean_d = sycl::malloc_device<float>(num_features, math::q);
	float* std_d  = sycl::malloc_device<float>(num_features, math::q);
	math::q.memset(mean_d, 0, num_features * sizeof(float));
	math::q.memset(std_d, 0, num_features * sizeof(float));

	float* data_p = data.elements;
	std::size_t mc = cols;
	std::size_t nf = num_features;

	// Sum per feature column (atomic into mean_d).
	math::q.parallel_for(sycl::range<2>(rows, nf), [=](sycl::id<2> idx) {
		std::size_t i = idx[0];
		std::size_t j = idx[1];
		sycl::atomic_ref<float,
						 sycl::memory_order::relaxed,
						 sycl::memory_scope::device,
						 sycl::access::address_space::global_space> ref(mean_d[j]);
		ref.fetch_add(data_p[i * mc + j]);
	});

	// Convert sum -> mean.
	float inv_rows = 1.0f / static_cast<float>(rows);
	math::q.parallel_for(sycl::range<1>(nf), [=](sycl::id<1> idx) {
		mean_d[idx[0]] *= inv_rows;
	});

	// Sum of squared deviations (atomic into std_d).
	math::q.parallel_for(sycl::range<2>(rows, nf), [=](sycl::id<2> idx) {
		std::size_t i = idx[0];
		std::size_t j = idx[1];
		float d = data_p[i * mc + j] - mean_d[j];
		sycl::atomic_ref<float,
						 sycl::memory_order::relaxed,
						 sycl::memory_scope::device,
						 sycl::access::address_space::global_space> ref(std_d[j]);
		ref.fetch_add(d * d);
	});

	// Convert sum-of-squares -> std_dev.
	math::q.parallel_for(sycl::range<1>(nf), [=](sycl::id<1> idx) {
		std_d[idx[0]] = sycl::sqrt(std_d[idx[0]] * inv_rows);
	});

	// Z-score normalize in place; skip columns with near-zero variance.
	math::q.parallel_for(sycl::range<2>(rows, nf), [=](sycl::id<2> idx) {
		std::size_t i = idx[0];
		std::size_t j = idx[1];
		float s = std_d[j];
		if (s >= 1e-8f)
		{
			data_p[i * mc + j] = (data_p[i * mc + j] - mean_d[j]) / s;
		}
	});

	math::q.wait();
	sycl::free(mean_d, math::q);
	sycl::free(std_d, math::q);
}

int main()
{
	// todo: add this hyperparams to a json file
	float lr = 0.01;
	size_t epochs = 100;
	size_t batch_size = 102'350;
	size_t input_size = 33;

	std::vector<conv1d::config> conv_configs = {
		{4, 3, conv1d::type::valid, 1},
		{4, 3, conv1d::type::valid, 1}
	};
	std::vector<activation::type> conv_activations = {
		activation::type::relu,
		activation::type::relu
	};
	std::vector<pooling::config> pool_configs = {
		{pooling::type::max, 2, 2},
		{pooling::type::max, 2, 2}
	};

	std::string filename = "nsl_kdd.csv";
	math::matrix<float> dataset = read_csv(filename);
	size_t dataset_cols = dataset.shape().second;
	assert(dataset_cols > input_size);
	size_t output_size = dataset_cols - input_size;
	normalize_features(dataset, input_size);

	std::vector<size_t> layers = {116, 16, 8, 4, output_size};

	std::vector<activation::type> activations = {
		activation::type::relu,
		activation::type::relu,
		activation::type::relu,
		output_size == 1 ? activation::type::sigmoid : activation::type::softmax
	};

	loss::type loss = output_size == 1 ? loss::type::bce : loss::type::ce;
	std::cout << "Running on: "
			  << math::q.get_device().get_info<sycl::info::device::name>()
			  << std::endl;
	nn::counter c;
	nn::dense_layer dense(epochs,
						  lr,
						  input_size,
						  conv_configs,
						  conv_activations,
						  pool_configs,
						  layers,
						  activations,
						  loss,
						  dataset);
 	dense.train(batch_size, c);

	math::matrix predictions = dense.feedforward(nn::get_cols(dataset, 0, input_size));
	std::cout << "Accuracy  = " << std::round(dense.accuracy(predictions) * 100) << "%" << "\n";
	std::cout << "Precision = " << std::round(dense.precision(predictions) * 100) << "%" << "\n";
	std::cout << "F1-score  = " << std::round(dense.f1_score(predictions) * 100) << "%" << "\n";

	return 0;
}
