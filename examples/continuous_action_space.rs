//! An example solving the mountaincar environment in a continuous action space.
//! This example uses a single environment.
//!
//! Since `gym-rs` doesn't have a continuous mountain car env, it's implemented here.

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::var_builder::VarBuilderArgs;
use candle_nn::{Init, Linear, Optimizer, VarMap};
use candle_optimisers::adam::{Adam, ParamsAdam};
use ppo2::distribution::{DiagGaussianPD, Distribution};
use ppo2::{ActorCriticModel, PPO2, PPO2Config};
use rand::distributions::uniform::{UniformFloat, UniformSampler};
use rand::thread_rng;

/// Neural network with two hidden layers, using tanh as the activation function.
/// Has two heads, one for the actor and one for the critic.
struct Model {
	l1: Linear,
	l2: Linear,
	actor: Linear,
	critic: Linear,
	/// Logstd for the probability distribution
	logstd: Tensor,
}
impl ActorCriticModel for Model {
	/// [position, velocity]
	type ObsType = [f64; 2];
	/// < 0 = left, 0 = no change, > 0 = right
	type ActType = [f64; 1];

	fn run_tensor<const BATCH_SIZE: usize>(
		&mut self,
		obs: &[[f64; 2]; BATCH_SIZE],
	) -> candle_core::error::Result<(impl Distribution<[f64; 1]>, Tensor)> {
		// Output from the body
		let out = self
			.l2
			.forward(&self.l1.forward(&Tensor::new(obs, &Device::Cpu)?)?.tanh()?)?
			.tanh()?;

		Ok((
			DiagGaussianPD::new(self.actor.forward(&out)?, self.logstd.clone())?,
			self.critic.forward(&out)?.squeeze(1)?,
		))
	}
}

fn main() {
	const HIDDEN_LAYER_SIZE: usize = 128;

	// PPO2 will collect N_MINIBATCHES * MINIBATCH_SIZE steps of data before doing backward
	// strps/optimizing. When this happens, It will split the data into N_MINIBATCHES chunks of
	// size MINIBATCH_SIZE for processing.
	const N_MINIBATCHES: usize = 4;
	const MINIBATCH_SIZE: usize = 512;

	const N_ENVS: usize = 1;

	let mut var_map = VarMap::new();
	let var_builder_args =
		VarBuilderArgs::from_backend(Box::new(var_map.clone()), DType::F64, Device::Cpu);

	// Initialize model parameters
	let model = Model {
		l1: Linear::new(
			var_builder_args
				.get_with_hints(vec![HIDDEN_LAYER_SIZE, 2], "l1 w", Init::Uniform {
					lo: -1.,
					up: 1.,
				})
				.unwrap(),
			var_builder_args
				.get_with_hints(HIDDEN_LAYER_SIZE, "l1 b", Init::Uniform { lo: -1., up: 1. })
				.ok(),
		),
		l2: Linear::new(
			var_builder_args
				.get_with_hints(
					vec![HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE],
					"l2 w",
					Init::Uniform { lo: -1., up: 1. },
				)
				.unwrap(),
			var_builder_args
				.get_with_hints(HIDDEN_LAYER_SIZE, "l2 b", Init::Uniform { lo: -1., up: 1. })
				.ok(),
		),
		actor: Linear::new(
			var_builder_args
				.get_with_hints(vec![1, HIDDEN_LAYER_SIZE], "actor w", Init::Uniform {
					lo: -1.,
					up: 1.,
				})
				.unwrap(),
			var_builder_args
				.get_with_hints(1, "actor b", Init::Uniform { lo: -1., up: 1. })
				.ok(),
		),
		critic: Linear::new(
			var_builder_args
				.get_with_hints(vec![1, HIDDEN_LAYER_SIZE], "critic w", Init::Uniform {
					lo: -1.,
					up: 1.,
				})
				.unwrap(),
			var_builder_args
				.get_with_hints(1, "critic b", Init::Uniform { lo: -1., up: 1. })
				.ok(),
		),
		logstd: var_builder_args.get(vec![1, 1], "logstd").unwrap(),
	};

	let mut ppo = PPO2::<N_MINIBATCHES, MINIBATCH_SIZE, N_ENVS, _, _>::new(
		model,
		Adam::new(var_map.all_vars(), ParamsAdam {
			// These are the settings used by default in the openai baseline
			// but they're not necessarily the best for every environment.
			//
			// I've found ~7e-3 to be pretty good for mountaincar to begin with,
			// although it sometimes gets stuck in a local maxima
			//lr: 3e-4,
			lr: 7e-3,
			eps: 1e-5,
			..ParamsAdam::default()
		})
		.unwrap(),
		PPO2Config::default(),
	);

	// Attempt to load saved data if it exists
	let _ = var_map.load("continuous.safetensors");

	let mut thread_rng = thread_rng();
	let uniform_float = UniformFloat::<f64>::new(-0.6, -0.4);

	// Mountaincar env state
	let mut position = uniform_float.sample(&mut thread_rng);
	let mut velocity = 0.;

	let mut action = ppo.step([[position, velocity]], [0.], [false]).unwrap()[0][0];

	let mut i = 0;

	// Run 256 epochs of training
	for _ in 0..256 {
		for _ in 0..N_MINIBATCHES * MINIBATCH_SIZE {
			// Do the mountaincar logic
			velocity = (velocity + action.clamp(-1., 1.) * 0.001 + (3. * position).cos() * -0.0025)
				.clamp(-0.07, 0.07);

			position = (position + velocity).max(-1.2);

			if position == -1.2 && velocity < 0. {
				velocity = 0.;
			}

			// Reset the env if it finished or the time limit was reached
			let done = position >= 0.5 || i == 100;

			// We're sort of cheating here to make it learn faster,
			// the standard mountaincar environment always uses a reward of -1.
			// Instead, we reward it for getting higher
			let reward = if position >= 0.5 {
				100.
			} else {
				// e^height
				(3. * position).sin().exp()
			};

			if done {
				println!(
					"Env {} at step {i}. Reward: {reward}",
					if position >= 0.5 {
						"completed"
					} else {
						"timed out"
					}
				);

				position = uniform_float.sample(&mut thread_rng);
				velocity = 0.;

				i = 0;
			} else {
				i += 1;
			}

			action = ppo.step([[position, velocity]], [reward], [done]).unwrap()[0][0];
		}

		// Optionally decay the learning rate
		//ppo.optimizer.set_learning_rate(/* something */);
	}

	// Save the current state of the model
	var_map.save("continuous.safetensors").unwrap();
}
