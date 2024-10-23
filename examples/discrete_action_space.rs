//! An example solving the mountaincar environment in a discrete action space.
//! This example uses a single environment

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::var_builder::VarBuilderArgs;
use candle_nn::{Init, Linear, Optimizer, VarMap};
use candle_optimisers::adam::{Adam, ParamsAdam};
use gym_rs::core::Env;
use gym_rs::envs::classical_control::mountain_car::MountainCarEnv;
use gym_rs::utils::renderer::RenderMode;
use ppo2::distribution::{CategoricalPD, Distribution};
use ppo2::{ActorCriticModel, PPO2, PPO2Config};

/// Neural network with two hidden layers, using tanh as the activation function.
/// Has two heads, one for the actor and one for the critic.
struct Model {
	l1: Linear,
	l2: Linear,
	actor: Linear,
	critic: Linear,
}
impl ActorCriticModel for Model {
	/// [position, velocity]
	type ObsType = [f64; 2];
	/// 0 = left, 1 = no change, 2 = right
	type ActType = u32;

	fn run_tensor<const BATCH_SIZE: usize>(
		&mut self,
		obs: &[[f64; 2]; BATCH_SIZE],
	) -> candle_core::error::Result<(impl Distribution<u32>, Tensor)> {
		const ACTION_SPACE_SIZE: usize = 3;

		// Output from the body
		let out = self
			.l2
			.forward(&self.l1.forward(&Tensor::new(obs, &Device::Cpu)?)?.tanh()?)?
			.tanh()?;

		Ok((
			CategoricalPD::<ACTION_SPACE_SIZE> {
				logits: self.actor.forward(&out)?,
			},
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
				.get_with_hints(vec![3, HIDDEN_LAYER_SIZE], "actor w", Init::Uniform {
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
	};

	let mut ppo = PPO2::<N_MINIBATCHES, MINIBATCH_SIZE, N_ENVS, _, _>::new(
		model,
		Adam::new(var_map.all_vars(), ParamsAdam {
			// these are the settings used by default in the openai baseline
			lr: 3e-4,
			eps: 1e-5,
			..ParamsAdam::default()
		})
		.unwrap(),
		PPO2Config::default(),
	);

	// Attempt to load saved data if it exists
	let _ = var_map.load("discrete.safetensors");

	let mut env = MountainCarEnv::new(RenderMode::None);

	let obs = env.reset(None, false, None).0;
	let mut action = ppo
		.step([[obs.position.0, obs.velocity.0]], [0.], [false])
		.unwrap()[0];

	let mut i = 0;

	// Run 64 training epochs
	for _ in 0..64 {
		for _ in 0..N_MINIBATCHES * MINIBATCH_SIZE {
			let mut state = env.step(action as usize);

			// Reset the env if it finished or time limit was reached
			if state.done || i == 100 {
				println!(
					"Env {} at step {i}",
					if state.done { "completed" } else { "timed out" }
				);
				state.observation = env.reset(None, false, None).0;

				i = 0;
			} else {
				i += 1;
			}

			action = ppo
				.step(
					[[state.observation.position.0, state.observation.velocity.0]],
					[state.reward.0],
					[state.done],
				)
				.unwrap()[0];
		}

		// Optionally decay the learning rate
		//ppo.optimizer.set_learning_rate(/* something */);
	}

	// Save the current state of the model
	var_map.save("discrete.safetensors").unwrap();
}
