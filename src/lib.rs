//! An implementation of the ppo2 reinforcement learning algorithm in rust.
//! This is ported fairly directly from [openai's baseline](https://github.com/openai/baselines).
//!
//! For usage examples, check under `examples/`.
//!
//! # Performance
//! If this is slow for you, firstly make sure you've enabled optimizations (`--release`).
//!
//! If it's still slow, try making your neural network smaller, or increasing
//! `N_MINIBATCHES`/`MINIBATCH_SIZE` to make the backwards step happen less often.
//!
//! You might also find, especially with small models, that this is sometimes faster on the cpu.

// `generic_const_exprs` is incomplete, but works here
#![allow(incomplete_features)]
#![feature(
	array_try_from_fn,
	generic_const_exprs,
	iterator_try_reduce,
	slice_as_chunks
)]

use core::array::from_fn;
use core::mem::{MaybeUninit, drop};
use core::ptr::{drop_in_place, write};

use candle_core::Tensor;
use candle_nn::Optimizer;
use rand::seq::SliceRandom;
use rand::thread_rng;

pub mod distribution;
use distribution::Distribution;
pub mod utils;
use utils::{cast_tensor_data, collect_minibatch, write_frame};

/// An actor/critic model. The actor model returns a probability distribution and critic returns an
/// array of values. For information on probability distributions, and some implementations, see the
/// [distribution] module. Using a single function allows for them to efficiently be two heads of
/// the same network.
///
/// NOTE: tensors returned must have a dtype of f64.
pub trait ActorCriticModel {
	type ObsType;
	type ActType;

	/// Run the models, outputting a single tensor for values.
	/// This is called directly when actually training, and by the default
	/// [`ActorCriticModel::run_array`] implementation.
	fn run_tensor<const BATCH_SIZE: usize>(
		&mut self,
		obs: &[Self::ObsType; BATCH_SIZE],
	) -> candle_core::Result<(impl Distribution<Self::ActType>, Tensor)>;

	/// Run the models returning an array as output.
	/// This version is called by [`PPO2::step`] while collecting samples, and is directly passed
	/// the observation passed into `step()`.
	///
	/// The standard implementation just calls [`ActorCriticModel::run_tensor`],
	/// converting the output.
	///
	/// If it's efficient for your model to return an array compared to a `Tensor`,
	/// reimplementing this could increase performance.
	fn run_array<const N_ENVS: usize>(
		&mut self,
		obs: &[Self::ObsType; N_ENVS],
	) -> candle_core::Result<(impl Distribution<Self::ActType>, [f64; N_ENVS])> {
		self.run_tensor(obs).and_then(|(pd, values)| {
			cast_tensor_data(&values).map(|values_array| (pd, values_array))
		})
	}
}

/// Config for a PPO2 agent, because there are a lot of parameters. Usage:
/// ```
/// let ppo = PPO2::new(
///   model,
///   optimizer
///   PPO2Builder {
///     something: 2.5,
///     // other options you want to change from defaults
///     ...PPO2Builder::default()
///   }
/// );
/// ```
///
/// Or just `PPO2::new(.., PPO2Builder::default())` to use the default options
pub struct PPO2Config {
	/// Policy entropy coefficient in the optimization objective
	pub ent_coef: f64,
	/// Value function loss coefficient in the optimization objective
	pub vf_coef: f64,
	/// Discounting factor
	pub gamma: f64,
	/// Advantage estimation discounting factor (lambda in the paper)
	pub lam: f64,
	/// Number of training epochs per update
	pub noptepochs: u8,
	/// Clipping range. If you want to change this during training, you can change it directly on
	/// the `PPO2` object before stepping
	pub cliprange: f64,
}
impl Default for PPO2Config {
	fn default() -> PPO2Config {
		PPO2Config {
			ent_coef: 0.,
			vf_coef: 0.5,
			gamma: 0.99,
			lam: 0.95,
			noptepochs: 4,
			cliprange: 0.2,
		}
	}
}

/// A struct for training a neural network using the PPO2 algorithm.
///
/// You can create an instance using [`Self::new`].
///
/// This contains 6 buffers of size `N_MINIBATCHES * MINIBATCH_SIZE * N_ENVS`, which are allocated
/// on the stack. These may be to big to store on your stack, so you may want to create it on the
/// heap with [`Self::new_boxed`].
///
/// You still need to be careful about making the buffers too big, since [`Self::step`] creates
/// another temporary one on the stack, and large buffer sizes won't necessarily improve training.
pub struct PPO2<
	const N_MINIBATCHES: usize,
	const MINIBATCH_SIZE: usize,
	const N_ENVS: usize,
	M: ActorCriticModel,
	O: Optimizer,
> where
	[(); N_ENVS * MINIBATCH_SIZE * N_MINIBATCHES]:,
{
	config: PPO2Config,

	model: M,
	pub optimizer: O,

	/// Current position building the buffers
	step: usize,

	// Buffers being built with `PPO::step`
	mb_rewards: [MaybeUninit<f64>; N_ENVS * MINIBATCH_SIZE * N_MINIBATCHES],
	mb_obs: [MaybeUninit<M::ObsType>; N_ENVS * MINIBATCH_SIZE * N_MINIBATCHES],
	mb_actions: [MaybeUninit<M::ActType>; N_ENVS * MINIBATCH_SIZE * N_MINIBATCHES],
	mb_values: [MaybeUninit<f64>; N_ENVS * MINIBATCH_SIZE * N_MINIBATCHES],
	mb_neglogpacs: [MaybeUninit<f64>; N_ENVS * MINIBATCH_SIZE * N_MINIBATCHES],
	mb_dones: [MaybeUninit<bool>; N_ENVS * MINIBATCH_SIZE * N_MINIBATCHES],
}
impl<
	const N_MINIBATCHES: usize,
	const MINIBATCH_SIZE: usize,
	const N_ENVS: usize,
	M: ActorCriticModel,
	O: Optimizer,
> PPO2<N_MINIBATCHES, MINIBATCH_SIZE, N_ENVS, M, O>
where
	M::ObsType: Clone,
	M::ActType: Clone,
	[(); N_ENVS * MINIBATCH_SIZE * N_MINIBATCHES]:,
	[(); N_ENVS * MINIBATCH_SIZE]:,
{
	/// Create the [`PPO2`] instance. The model is your own struct implementing the
	/// [`ActorCriticModel`] trait, and the optimizer can be any [`candle_nn::optim::Optimizer`],
	/// although the openai PPO2 baseline uses adam, with a default learning rate starting at 3e-4
	/// and epsilon of 1e-5.
	///
	/// The optimizer must know about everything that you want to train, i.e. your neural networks'
	/// layers, and if you're using the [`distribution::DiagGaussianPD`] probability distribution
	/// you might want to also train the logstd parameter.
	///
	/// If you want to change the learning rate, or other optimizer parameters while training,
	/// you can modify the optimizer directly on the [`PPO2`] struct.
	pub fn new(model: M, optimizer: O, config: PPO2Config) -> Self {
		PPO2 {
			config,

			model,
			optimizer,

			step: 0,

			mb_rewards: [const { MaybeUninit::uninit() }; N_ENVS * MINIBATCH_SIZE * N_MINIBATCHES],
			mb_obs: [const { MaybeUninit::uninit() }; N_ENVS * MINIBATCH_SIZE * N_MINIBATCHES],
			mb_actions: [const { MaybeUninit::uninit() }; N_ENVS * MINIBATCH_SIZE * N_MINIBATCHES],
			mb_values: [const { MaybeUninit::uninit() }; N_ENVS * MINIBATCH_SIZE * N_MINIBATCHES],
			mb_neglogpacs: [const { MaybeUninit::uninit() };
				N_ENVS * MINIBATCH_SIZE * N_MINIBATCHES],
			mb_dones: [const { MaybeUninit::uninit() }; N_ENVS * MINIBATCH_SIZE * N_MINIBATCHES],
		}
	}

	/// Create a new `PPO2` instance directly on the heap, this is useful if you want massive batch
	/// sizes/many envs.
	///
	/// For more info, see documentation under [`Self::new`].
	pub fn new_boxed(model: M, optimizer: O, config: PPO2Config) -> Box<Self> {
		let mut ppo2 = Box::new_uninit();
		unsafe {
			let ppo2_ref = &mut *(&mut *ppo2 as *mut _ as *mut Self);

			write(&mut ppo2_ref.config as _, config);

			write(&mut ppo2_ref.model as _, model);
			write(&mut ppo2_ref.optimizer as _, optimizer);

			write(&mut ppo2_ref.step as *mut usize, 0);

			// Minibatches don't need to be set, since they start uninitialized

			ppo2.assume_init()
		}
	}

	/// Collect one step of data and return an action.
	///
	/// When the buffers are full (after `MINIBATCH_SIZE * N_MINIBATCHES` steps),
	/// this function will also train the model using the collected data.
	///
	/// # Panics
	/// This function will panic with index out of bounds if any of `N_ENVS`, `MINIBATCH_SIZE` or
	/// `N_MINIBATCHES` are 0
	pub fn step(
		&mut self,
		obs: [M::ObsType; N_ENVS],
		rewards: [f64; N_ENVS],
		dones: [bool; N_ENVS],
	) -> candle_core::Result<[M::ActType; N_ENVS]> {
		let (pd, values) = self.model.run_array(&obs)?;

		// Get the pd stuff out of the way

		// Take an action
		let actions = pd.sample()?;

		// Calculate the neg log of our probability
		let neglogpacs = pd.neglogp_array(&actions)?;

		// We need to drop pd here, since it could be borrowing data from self.model,
		// preventing us from using it later
		drop(pd);

		// The rewards are meaningless on the first step
		if self.step != 0 {
			// Write the rewards to the previous step
			write_frame(&mut self.mb_rewards, self.step - N_ENVS, rewards);
		}

		// If we've filled our buffers, we can train
		if self.step == N_ENVS * MINIBATCH_SIZE * N_MINIBATCHES {
			self.step = 0;

			let mut mb_advs =
				[MaybeUninit::<f64>::uninit(); N_ENVS * MINIBATCH_SIZE * N_MINIBATCHES];

			// This is safe, because we know all buffers have been initialized
			unsafe {
				// GAE (advantage estimation)
				let mut lastgaelam = from_fn::<_, N_ENVS, _>(|i| {
					let x = self.mb_rewards[N_ENVS * (MINIBATCH_SIZE * N_MINIBATCHES - 1) + i]
						.assume_init_read()
						- self.mb_values[N_ENVS * (MINIBATCH_SIZE * N_MINIBATCHES - 1) + i]
							.assume_init_read();

					if dones[i] {
						x
					} else {
						x + self.config.gamma * values[i]
					}
				});

				write_frame(
					&mut mb_advs,
					N_ENVS * (MINIBATCH_SIZE * N_MINIBATCHES - 1),
					lastgaelam,
				);

				for step in (0..N_ENVS * (MINIBATCH_SIZE * N_MINIBATCHES - 1))
					.step_by(N_ENVS)
					.rev()
				{
					lastgaelam = from_fn(|i| {
						let x = self.mb_rewards[step + i].assume_init_read()
							- self.mb_values[step + i].assume_init_read();

						if self.mb_dones[step + N_ENVS + i].assume_init_read() {
							x
						} else {
							x + self.config.gamma
								* (self.mb_values[step + N_ENVS + i].assume_init_read()
									+ self.config.lam * lastgaelam[i])
						}
					});

					write_frame(&mut mb_advs, step, lastgaelam);
				}
			}

			// Create an array of indices into the minibuffers to process, shuffled at each epoch
			let mut inds = from_fn::<_, { N_ENVS * MINIBATCH_SIZE * N_MINIBATCHES }, _>(|i| i);

			for _ in 0..self.config.noptepochs {
				inds.shuffle(&mut thread_rng());

				// Iterate through the batch in chunks
				// It's safe to split the buffers like this, since N_ENVS * MINIBATCH_SIZE is a
				// factor of the buffer size
				for mbinds in unsafe { inds.as_chunks_unchecked::<{ N_ENVS * MINIBATCH_SIZE }>() } {
					let obs = collect_minibatch(&self.mb_obs, mbinds);
					let (pd, current_values) = self.model.run_tensor(&obs)?;

					let old_values = Tensor::new(
						&collect_minibatch(&self.mb_values, mbinds),
						current_values.device(),
					)?;

					// Difference between the old and current predicted values
					let vf_diff = (current_values - old_values)?;

					let advs = Tensor::new(&collect_minibatch(&mb_advs, mbinds), vf_diff.device())?;

					let advs_normalized = ((&advs - advs.mean_all()?.to_scalar::<f64>()?)?
						/ (advs.var(0)?.to_scalar::<f64>()?.sqrt() + 1e-8))?;

					// Calculate ratio (pi current policy / pi old policy)
					let ratio = (Tensor::new(
						&collect_minibatch(&self.mb_neglogpacs, mbinds),
						advs_normalized.device(),
					)? - pd
						.neglogp_tensor(&collect_minibatch(&self.mb_actions, mbinds))?)
					.unwrap()
					.exp()?;

					// total loss = value loss * vf_coef - (-policy gradient loss)
					let mut loss = ((0.5
						* (&vf_diff - &advs)?
							.sqr()?
							.maximum(
								&(vf_diff.clamp(-self.config.cliprange, self.config.cliprange)
									- advs)?
									.sqr()?,
							)?
							.mean_all()?)? * self.config.vf_coef
						- (&advs_normalized * &ratio)?
							.minimum(
								&(advs_normalized
									* ratio.clamp(
										1. - self.config.cliprange,
										1. + self.config.cliprange,
									))?,
							)?
							.mean_all()?)?;

					if self.config.ent_coef != 0. {
						loss = (loss - pd.mean_entropy()? * self.config.ent_coef)?;
					}

					// TODO: could implement max_grad_norm here, but we'd need to be able to iterate
					// the gradients (candle doesn't currently allow this)

					self.optimizer.step(&loss.backward()?)?;
				}
			}

			// Drop all of the potential tensors in the minifuffers (we need to do this manually
			// because we're using MaybeUninit) We know all minibuffers have been filled, so it's
			// safe to drop all of them
			unsafe {
				drop_in_place(
					&mut self.mb_obs as *mut _
						as *mut [M::ObsType; N_ENVS * MINIBATCH_SIZE * N_MINIBATCHES],
				);
				drop_in_place(
					&mut self.mb_actions as *mut _
						as *mut [M::ActType; N_ENVS * MINIBATCH_SIZE * N_MINIBATCHES],
				);
			}
		}

		write_frame(&mut self.mb_values, self.step, values);
		write_frame(&mut self.mb_neglogpacs, self.step, neglogpacs);
		write_frame(&mut self.mb_dones, self.step, dones);
		write_frame(&mut self.mb_obs, self.step, obs);
		write_frame(&mut self.mb_actions, self.step, actions.clone());

		self.step += N_ENVS;

		Ok(actions)
	}
}
impl<
	const N_MINIBATCHES: usize,
	const MINIBATCH_SIZE: usize,
	const N_ENVS: usize,
	M: ActorCriticModel,
	O: Optimizer,
> Drop for PPO2<N_MINIBATCHES, MINIBATCH_SIZE, N_ENVS, M, O>
where
	[(); N_ENVS * MINIBATCH_SIZE * N_MINIBATCHES]:,
{
	fn drop(&mut self) {
		unsafe {
			drop_in_place(&mut self.mb_obs[..self.step] as *mut _ as *mut [M::ObsType]);
			drop_in_place(&mut self.mb_actions[..self.step] as *mut _ as *mut [M::ActType]);
		}
	}
}
