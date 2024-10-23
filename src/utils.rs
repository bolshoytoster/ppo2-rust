use core::array::from_fn;
use core::ptr::write;
use std::mem::MaybeUninit;

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, Storage, Tensor};

/// Write frame data to a buffer at the given step.
/// Doesn't check to make sure the data will fit, so could segfault if used incorrectly.
pub(crate) fn write_frame<const M: usize, const N: usize, T>(
	buffer: &mut [MaybeUninit<T>; M],
	step: usize,
	data: [T; N],
) {
	unsafe { write(&mut buffer[step] as *mut _ as *mut _, data) };
}

/// Helper function for collecting the minibatches from their indices
/// Assumes the data is initialized
pub(crate) fn collect_minibatch<const BATCH_SIZE: usize, const MINIBATCH_SIZE: usize, T: Clone>(
	data: &[MaybeUninit<T>; BATCH_SIZE],
	indices: &[usize; MINIBATCH_SIZE],
) -> [T; MINIBATCH_SIZE] {
	from_fn(|i| (*unsafe { data[indices[i]].assume_init_ref() }).clone())
}

// TODO: should be pub(crate)

/// Convert a tensor's data to a given type, not checking if they are the same size.
///
/// # Errors
/// This will return an error if the tensor's DType isn't f64
pub fn cast_tensor_data<T: Clone>(x: &Tensor) -> candle_core::Result<T> {
	map_storage(x, |cpu_storage| {
		if let CpuStorage::F64(vec) = cpu_storage {
			Ok(unsafe { (*(&**vec as *const _ as *const T)).clone() })
		} else {
			Err(candle_core::Error::UnsupportedDTypeForOp(
				x.dtype(),
				"distribution (must be f64)",
			))
		}
	})
}

/// Run a function on the given tensor's raw data,
/// copying the tensor to main memory if it isn't there already.
pub(crate) fn map_storage<T>(
	tensor: &Tensor,
	f: impl FnOnce(&CpuStorage) -> candle_core::Result<T>,
) -> candle_core::Result<T> {
	match *tensor.storage_and_layout().0 {
		Storage::Cpu(ref cpu_storage) => f(cpu_storage),
		Storage::Cuda(ref cuda_storage) => cuda_storage
			.to_cpu_storage()
			.and_then(|cpu_storage| f(&cpu_storage)),
		Storage::Metal(ref metal_storage) => metal_storage
			.to_cpu_storage()
			.and_then(|cpu_storage| f(&cpu_storage)),
	}
}
