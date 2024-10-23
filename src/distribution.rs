//! Probability distributions.
//!
//! If your action space is discrete, use [`CategoricalPD`], which returns a `u32`.
//! If your action space is continuous, use [`DiagGaussianPD`], which returns a `[f64;
//! ACT_SPACE_SIZE]`.

use core::array::{from_fn, try_from_fn};
use core::cmp::Ordering;
use core::f64::consts::TAU;

use candle_core::{CpuStorage, Tensor};

use crate::utils::{cast_tensor_data, map_storage};

/// A probability distribution
pub trait Distribution<ActType> {
	fn sample<const BATCH_SIZE: usize>(&self) -> candle_core::Result<[ActType; BATCH_SIZE]>;

	/// Compute the negative log probabilities of the given actions.
	/// Input is the same as the output of [`sample`].
	/// Output is a 1D tensor, with length `BATCH_SIZE`.
	///
	/// This is called by `PPO2::step` when training.
	fn neglogp_tensor<const BATCH_SIZE: usize>(
		&self,
		x: &[ActType; BATCH_SIZE],
	) -> candle_core::Result<Tensor>;

	/// Similar to [`Distribution::neglogp_tensor`], but returns an array instead.
	/// This is used when collecting samples.
	///
	/// The default implementation just calls `neglogp_tensor` and converts the output.
	///
	/// If your distribution can do this more efficiently than the default implementation,
	/// you can reimplement this yourself.
	fn neglogp_array<const BATCH_SIZE: usize>(
		&self,
		x: &[ActType; BATCH_SIZE],
	) -> candle_core::Result<[f64; BATCH_SIZE]> {
		self.neglogp_tensor(x)
			.and_then(|neglogp| cast_tensor_data(&neglogp))
	}

	/// Compute the entropy of this distribution, then get the mean value as a tensor with one item.
	fn mean_entropy(&self) -> candle_core::error::Result<Tensor>;
}

/// A gaussian probability distribution.
/// You probably want to use this if your action space is continuous.
///
/// Sampling from this can segfault if the dimensions of `logstd` and `mean` don't match
/// the expected batch and action size.
pub struct DiagGaussianPD {
	logstd: Tensor,
	stretched_logstd: Tensor,
	mean: Tensor,
}
impl DiagGaussianPD {
	/// Create a gaussian probability distribution.
	///
	/// This distribution requires a `logstd` tensor, which should be 2D, with size 1 x
	/// `action_space_size` You might want to train the `logstd` like in the openai baseline, if
	/// so, just create it with the [`candle_nn::var_builder::VarBuilderArgs`] that you pass to
	/// your optimizer: ```
	/// let logstd = var_builder_args.get((1 /* action space size */,), "logstd");
	///
	/// // Later, probably in your ActorCriticModel implementation
	/// let pd = DiagGaussianPD::new(policy_network_output, logstd.clone());
	/// ```
	pub fn new(mean: Tensor, logstd: Tensor) -> candle_core::Result<DiagGaussianPD> {
		// Stretch logstd to be the same size as mean
		logstd
			.broadcast_as(mean.shape())
			.map(|stretched_logstd| DiagGaussianPD {
				logstd,
				stretched_logstd,
				mean,
			})
	}
}

impl<const ACT_SIZE: usize> Distribution<[f64; ACT_SIZE]> for DiagGaussianPD {
	fn sample<const BATCH_SIZE: usize>(
		&self,
	) -> candle_core::Result<[[f64; ACT_SIZE]; BATCH_SIZE]> {
		cast_tensor_data(
			&(&self.mean
				+ (self.stretched_logstd.exp()? * self.stretched_logstd.randn_like(0., 1.))?)?,
		)
	}

	fn neglogp_tensor<const BATCH_SIZE: usize>(
		&self,
		x: &[[f64; ACT_SIZE]; BATCH_SIZE],
	) -> candle_core::Result<Tensor> {
		self.stretched_logstd.sum(1)?
			+ 0.5
				* (((Tensor::new(x, self.mean.device())? - &self.mean)?
					/ self.stretched_logstd.exp())?
				.sum(1)?
				.sqr()? + TAU.ln() * ACT_SIZE as f64)?
	}

	fn mean_entropy(&self) -> candle_core::Result<Tensor> {
		self.logstd
			.sum(vec![0, 1])
			.and_then(|sum| sum + (0.5 * TAU.ln() + 0.5) * ACT_SIZE as f64)
	}
}

/// A categorical probability distribution,
/// you should use this if you have a discrete action space.
///
/// The type parameter `ACT_SIZE` is the action space size (number of possible actions).
pub struct CategoricalPD<const ACT_SIZE: usize> {
	pub logits: Tensor,
}
impl<const ACT_SIZE: usize> Distribution<u32> for CategoricalPD<ACT_SIZE> {
	fn sample<const BATCH_SIZE: usize>(&self) -> candle_core::Result<[u32; BATCH_SIZE]> {
		self.logits
			.rand_like(0., 1.)
			.and_then(|x| x.log())
			.and_then(|x| x.neg())
			.and_then(|x| &self.logits - x.log())
			.and_then(|x| x.argmax_keepdim(1))
			.and_then(|x| {
				map_storage(&x, |cpu_storage| {
					// Tensor::argmax_keepdim should always return a U32 tensor
					let CpuStorage::U32(vec) = cpu_storage else {
						unreachable!()
					};

					Ok(unsafe { *(vec as *const _ as *const _) })
				})
			})
	}

	// This isn't very efficient, but we need the gradients
	fn neglogp_tensor<const BATCH_SIZE: usize>(
		&self,
		x: &[u32; BATCH_SIZE],
	) -> candle_core::Result<Tensor> {
		// logits - max_logits.
		let backprop = (&self.logits
			- self
				.logits
				.max_keepdim(1)?
				.broadcast_as(self.logits.shape()))?;

		//  sum(-labels *
		//     ((logits - max_logits) - log(sum(exp(logits - max_logits)))))
		//  along classes
		(Tensor::new(
			// We do one hot encoding ourselves, since it's faster than candle-nn's implementation
			&x.map(|i| from_fn::<_, ACT_SIZE, _>(|j| if i == j as u32 { 1. } else { 0. })),
			backprop.device(),
		)? * (backprop
			.exp()?
			.sum_keepdim(1)?
			.log()?
			.broadcast_as(self.logits.shape())
			- backprop))?
			.sum_keepdim(1)?
			.squeeze(1)
	}

	// Reimplement neglogp_array, since we can do it more efficiently without tensors
	fn neglogp_array<const BATCH_SIZE: usize>(
		&self,
		x: &[u32; BATCH_SIZE],
	) -> candle_core::Result<[f64; BATCH_SIZE]> {
		let logits = cast_tensor_data::<[[f64; ACT_SIZE]; BATCH_SIZE]>(&self.logits)?;

		try_from_fn(|i| {
			// Can't just use Iterator::max_by, since we want to handle NaNs
			match logits[i].iter().try_reduce(|a, b| {
				a.partial_cmp(b)
					.map(|ordering| if ordering == Ordering::Less { b } else { a })
			}) {
				Some(Some(max)) => Ok(logits[i].iter().map(|x| (x - max).exp()).sum::<f64>().ln()
					- logits[i][x[i] as usize]
					+ max),
				Some(None) => Err(candle_core::Error::Msg(
					"CategoricalPD's logits contained a NaN".to_owned(),
				)),
				None => Err(candle_core::Error::Msg(
					"CategoricalPD's ACT_SIZE was 0".to_owned(),
				)),
			}
		})
	}

	fn mean_entropy(&self) -> candle_core::Result<Tensor> {
		let a0 = (&self.logits
			- self
				.logits
				.max_keepdim(1)?
				.broadcast_as(self.logits.shape()))?;
		let ea0 = a0.exp()?;
		let z0 = ea0.sum_keepdim(1)?.broadcast_as(self.logits.shape())?;
		((ea0 / &z0)? * (z0.log()? - a0))?.sum_all()? / self.logits.dims()[0] as f64
	}
}
