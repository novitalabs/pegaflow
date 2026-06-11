use libc::{CPU_SET, CPU_ZERO, cpu_set_t, pthread_self, pthread_setaffinity_np, sched_getcpu};
use syscalls::Errno;

#[inline]
pub(crate) fn current_tid_and_cpu() -> (i64, i32) {
    unsafe {
        let tid = libc::syscall(libc::SYS_gettid) as i64;
        let cpu = sched_getcpu();
        (tid, cpu)
    }
}

/// Restrict the current thread to the given CPU set. A single CPU pins the
/// thread to that core; multiple CPUs (e.g. one NUMA node's cores) keep the
/// thread NUMA-local while letting the scheduler avoid busy cores.
pub(crate) fn pin_cpus(cpus: &[u16]) -> Result<(), Errno> {
    assert!(!cpus.is_empty());
    unsafe {
        let (tid, cpu_before) = current_tid_and_cpu();
        let mut cpuset = std::mem::zeroed();
        CPU_ZERO(&mut cpuset);
        for &cpu in cpus {
            CPU_SET(cpu as usize, &mut cpuset);
        }
        let ret = pthread_setaffinity_np(pthread_self(), size_of::<cpu_set_t>(), &cpuset);
        if ret != 0 {
            return Err(Errno::new(ret));
        }
        let (_, cpu_after) = current_tid_and_cpu();
        log::debug!(
            "[pin_cpus] tid={} target_cpus={:?} sched_cpu_before={} sched_cpu_after={}",
            tid,
            cpus,
            cpu_before,
            cpu_after
        );
        Ok(())
    }
}
